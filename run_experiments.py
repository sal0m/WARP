import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import copy
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)

def calc_returns(rewards, gamma=0.99):
    rets = []
    r_sum = 0
    for r in reversed(rewards):
        r_sum = r + gamma * r_sum
        rets.insert(0, r_sum)
    rets = torch.tensor(rets)
    # нормализация чтоб градиенты не взрывались
    rets = (rets - rets.mean()) / (rets.std() + 1e-8)
    return rets

def get_kl(pol, anch, s):
    with torch.no_grad():
        a_probs = anch(s)
    p_probs = pol(s)
    kl = torch.sum(p_probs * (torch.log(p_probs + 1e-8) - torch.log(a_probs + 1e-8)))
    return kl.item()

def eval_pol(env_name, pol, eps=10):
    env = gym.make(env_name)
    total = []
    for _ in range(eps):
        s, _ = env.reset()
        done = False
        ep_r = 0
        while not done:
            st = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                probs = pol(st)
            m = Categorical(probs)
            a = m.sample().item()
            s, r, term, trunc, _ = env.step(a)
            done = term or trunc
            ep_r += r
        total.append(ep_r)
    env.close()
    return np.mean(total)

def train_base(env_name, s_dim, a_dim, eps=100):
    env = gym.make(env_name)
    pol = PolicyNetwork(s_dim, a_dim)
    opt = optim.Adam(pol.parameters(), lr=1e-2)
    # чутка проучим чтоб не совсем с нуля
    for _ in range(eps):
        s, _ = env.reset()
        done = False
        logp, rs = [], []
        while not done:
            st = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
            probs = pol(st)
            m = Categorical(probs)
            a = m.sample()
            logp.append(m.log_prob(a))
            s, r, term, trunc, _ = env.step(a.item())
            done = term or trunc
            rs.append(r)
        
        rets = calc_returns(rs)
        loss = sum([-lp * R for lp, R in zip(logp, rets)])
        
        opt.zero_grad()
        loss.backward()
        opt.step()
    env.close()
    return pol

def rl_loop(env_name, init_pol, sft_pol, anchor_type='ema', beta=0.1, mu=0.01, eps=200):
    env = gym.make(env_name)
    pol = copy.deepcopy(init_pol)
    anch = copy.deepcopy(init_pol if anchor_type == 'ema' else sft_pol)
    opt = optim.Adam(pol.parameters(), lr=5e-3)
    
    r_hist = []
    for _ in range(eps):
        s, _ = env.reset()
        done = False
        logp, rs, kl_pens = [], [], []
        
        while not done:
            st = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
            probs = pol(st)
            m = Categorical(probs)
            a = m.sample()
            logp.append(m.log_prob(a))
            
            with torch.no_grad():
                a_probs = anch(st)
                
            kl_p = torch.log(probs[0, a] + 1e-8) - torch.log(a_probs[0, a] + 1e-8)
            kl_pens.append(kl_p.item())
            
            s, r, term, trunc, _ = env.step(a.item())
            done = term or trunc
            rs.append(r)
        
        # r - beta * kl как в статье
        adj_rs = [r - beta * kp for r, kp in zip(rs, kl_pens)]
            
        rets = calc_returns(adj_rs)
        loss = sum([-lp * R for lp, R in zip(logp, rets)])
            
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if anchor_type == 'ema':
            # обновляем якорь по экспоненте
            with torch.no_grad():
                for p, ap in zip(pol.parameters(), anch.parameters()):
                    ap.data.copy_((1 - mu) * ap.data + mu * p.data)
                    
        r_hist.append(np.sum(rs))
        
    env.close()
    return pol, r_hist

def lerp(w1, w2, lam):
    return (1 - lam) * w1 + lam * w2

def slerp(w1, w2, lam):
    w1_f, w2_f = w1.flatten(), w2.flatten()
    n1 = w1_f / (torch.norm(w1_f) + 1e-8)
    n2 = w2_f / (torch.norm(w2_f) + 1e-8)
    
    dot = torch.sum(n1 * n2).clamp(-1.0, 1.0)
    omega = torch.acos(dot)
    sin_o = torch.sin(omega)
    
    # если вектора слишком близки, делаем обычный lerp
    if sin_o < 1e-5:
        return lerp(w1_f, w2_f, lam).view(w1.shape)
        
    r1 = torch.sin((1 - lam) * omega) / sin_o
    r2 = torch.sin(lam * omega) / sin_o
    
    return (r1 * w1_f + r2 * w2_f).view(w1.shape)

def merge(m1, m2, fn, init_m, lam=0.5, use_tasks=True):
    merged = copy.deepcopy(m1)
    d1, d2 = m1.state_dict(), m2.state_dict()
    d_init = init_m.state_dict()
    out = merged.state_dict()
    
    for k in d1:
        if use_tasks:
            # слияние через таск-вектора (как в статье)
            t1 = d1[k] - d_init[k]
            t2 = d2[k] - d_init[k]
            out[k] = d_init[k] + fn(t1, t2, lam)
        else:
            out[k] = fn(d1[k], d2[k], lam)
            
    merged.load_state_dict(out)
    return merged

if __name__ == '__main__':
    env = "CartPole-v1"
    
    with open("metrics.log", "w", encoding="utf-8") as f:
        def log(msg):
            print(msg)
            f.write(str(msg) + "\n")
            
        # 1. сначала сделаем базовую модель
        torch.manual_seed(42)
        sft = train_base(env, 4, 2, 100)
        log(f"база (sft): {eval_pol(env, sft):.1f}")
        
        # эээээксп 1: ema против обычного sft якоря
        torch.manual_seed(10)
        _, r_sft = rl_loop(env, sft, sft, 'sft', 0.01, 0.01, 200)
        torch.manual_seed(10) 
        pol1, r_ema = rl_loop(env, sft, sft, 'ema', 0.01, 0.01, 200)
        
        log("\n--- этап 1 ---")
        log(f"sft якорь: {np.mean(r_sft[-20:]):.1f}")
        log(f"ema якорь: {np.mean(r_ema[-20:]):.1f}")
        
        # эксп 2: slerp и lerp
        torch.manual_seed(20)
        pol2, _ = rl_loop(env, sft, sft, 'ema', 0.01, 0.01, 200)
        
        log("\n--- этап 2 ---")
        log(f"модель 1: {eval_pol(env, pol1):.1f}")
        log(f"модель 2: {eval_pol(env, pol2):.1f}")
        
        lerp_m = merge(pol1, pol2, lerp, sft, 0.5, True)
        slerp_m = merge(pol1, pol2, slerp, sft, 0.5, True)
        
        log(f"lerp: {eval_pol(env, lerp_m, 20):.1f}")
        log(f"slerp: {eval_pol(env, slerp_m, 20):.1f}")
        
        # эксп 3: liti (тянем обратно к sft)
        log("\n--- этап 3 (liti) ---")
        for e in [0.0, 0.3, 0.5, 0.8, 1.0]:
            liti = merge(sft, slerp_m, lerp, sft, e, False)
            log(f"eta={e}: {eval_pol(env, liti, 20):.1f}")
        
        # итеративный процесс
        # берем liti с eta=0.5 как новый инит
        init2 = merge(sft, slerp_m, lerp, sft, 0.5, False)
        
        log("\n--- итеративный warp ---")
        torch.manual_seed(30)
        p3, _ = rl_loop(env, init2, sft, 'ema', 0.01, 0.01, 200)
        torch.manual_seed(40)
        p4, _ = rl_loop(env, init2, sft, 'ema', 0.01, 0.01, 200)
        
        sl2 = merge(p3, p4, slerp, init2, 0.5, True)
        liti2 = merge(sft, sl2, lerp, sft, 0.5, False)
        
        log(f"итерация 1: {eval_pol(env, init2, 20):.1f}")
        log(f"итерация 2: {eval_pol(env, liti2, 20):.1f}")
        log("\nвсе логи сохранены в metrics.log")
