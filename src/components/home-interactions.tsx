'use client';

import Link from 'next/link';
import { ArrowRight, BookOpen, Braces, Check, Cpu, Wrench } from 'lucide-react';
import { useState } from 'react';

const lifecycle = [
  {
    key: 'text',
    label: 'Text',
    title: '原始文本',
    token: '"你好"',
    note: '模型不直接看见「字」。先用确定的编码把 code point 变成 bytes，才有覆盖任意文本的有限字母表。',
  },
  {
    key: 'token',
    label: 'Tokenize',
    title: 'BPE 编码',
    token: '[6,372, 8,401]',
    note: '频繁相邻的 byte pair 被反复合并成新 token。merge 的顺序本身也定义了编码规则——冲突时先试先合并的 pair。',
  },
  {
    key: 'tensor',
    label: 'Forward',
    title: '前向传播',
    token: '[B,T,V] logits',
    note: 'token ids 经过 Embedding、Attention、MLP 变成词表上的分数。每层传递的是 [B,T,D] 张量，Attention 混合位置，MLP 改写道。',
  },
  {
    key: 'loss',
    label: 'Loss',
    title: '学习信号',
    token: 'CE = 2.31',
    note: '序列右移一位就是 target。交叉熵告诉模型应该抬高哪个 token 的概率——不需要人工标注类别。',
  },
  {
    key: 'system',
    label: 'System',
    title: '高效训练',
    token: '312 TFLOP/s',
    note: '同一数学结果被重新安排到 GPU 集群。fused kernel、FSDP 与 overlap 让昂贵硬件少等待，但不改变模型语义。',
  },
  {
    key: 'behavior',
    label: 'Align',
    title: '后训练',
    token: 'p(answer|q)',
    note: '后训练没有凭空增加「回答模块」。它仍然更新 next-token policy，只是数据来源、reward 与评测目标发生了变化。',
  },
] as const;

export function HeroConsole() {
  const [active, setActive] = useState(0);
  const stage = lifecycle[active];

  return (
    <div className="hero-console">
      <div className="console-chrome">
        <div><i /><i /><i /></div>
        <span>model_lifecycle / stage_{String(active + 1).padStart(2, '0')}</span>
        <b>TRACE</b>
      </div>
      <div className="console-body">
        <p>
          <strong>{stage.title}</strong> — {stage.note}
        </p>
      </div>
      <div style={{ padding: '8px 18px 0' }}>
        <div style={{
          display: 'inline-block',
          padding: '6px 10px',
          border: '1px solid var(--lab-line)',
          borderRadius: 6,
          fontFamily: 'var(--lab-mono)',
          fontSize: 11,
          color: 'var(--lab-ink)',
        }}>
          {stage.token}
        </div>
      </div>
      <div className="console-stages" role="tablist" aria-label="模型生命周期">
        {lifecycle.map((item, index) => (
          <button
            key={item.key}
            role="tab"
            aria-selected={index === active}
            onClick={() => setActive(index)}
          >
            {item.label}
          </button>
        ))}
      </div>
    </div>
  );
}

const paths = [
  {
    key: 'course',
    label: '第一次系统学习',
    icon: BookOpen,
    title: '先完成一条最小闭环',
    steps: ['读：先修知识与 Lectures 1–4', '写：A1 的 tokenizer 与 Transformer', '验：训练 TinyStories 并解释生成轨迹'],
    href: '/docs/foundations',
  },
  {
    key: 'assignment',
    label: '准备完成作业',
    icon: Wrench,
    title: '按题目依赖，而不是 PDF 页码写代码',
    steps: ['读：当前题目与最小心智模型', '写：完整参考实现并逐段复述', '验：官方测试、边界样例与失败症状'],
    href: '/docs/assignments',
  },
  {
    key: 'source',
    label: '想深入系统源码',
    icon: Cpu,
    title: '沿数据搬运路径读系统代码',
    steps: ['画：HBM ↔ SRAM ↔ Tensor Core', '测：benchmark 后再 profile', '改：保持数值 oracle，不破坏语义'],
    href: '/docs/systems',
  },
];

export function PathSelector() {
  const [active, setActive] = useState('course');
  const path = paths.find((item) => item.key === active) ?? paths[0];

  return (
    <div className="path-picker">
      <div className="path-tabs" role="tablist" aria-label="选择学习路线">
        {paths.map(({ key, label, icon: Icon }) => (
          <button key={key} role="tab" aria-selected={key === active} onClick={() => setActive(key)}>
            <Icon size={16} /> {label}
          </button>
        ))}
      </div>
      <div className="path-panel">
        <span>RECOMMENDED ROUTE</span>
        <h3>{path.title}</h3>
        <ol>
          {path.steps.map((step, index) => <li key={step}><b>{index + 1}</b>{step}</li>)}
        </ol>
        <Link href={path.href}>从这里出发 <ArrowRight size={16} /></Link>
      </div>
    </div>
  );
}

export function ProofLedger() {
  return (
    <div className="hero-proof" aria-label="课程特点">
      <span><Braces size={14} /> 公式旁边就是 shape</span>
      <span><Check size={14} /> 实现通过课程测试</span>
      <span><BookOpen size={14} /> 题面与讲义逐段对照</span>
    </div>
  );
}
