'use client';

import Link from 'next/link';
import { ArrowRight, BookOpen, Braces, Cpu, Wrench } from 'lucide-react';
import { useState } from 'react';

const consoleStages = [
  {
    key: 'tokenize',
    label: '① 文本切成 token',
    title: 'TOKENIZER / BYTE-LEVEL BPE',
    note: 'Unicode 先编码成 UTF-8 bytes，再按 BPE merge 顺序合并；任何文本因此都有表示。',
    tokens: [
      { piece: 'Once', id: 7912, width: 84, tone: 'lime' },
      { piece: ' upon', id: 2461, width: 62, tone: 'violet' },
      { piece: ' a', id: 257, width: 38, tone: 'orange' },
      { piece: ' time', id: 892, width: 52, tone: 'blue' },
    ],
    target: '输入形状 [B, T] = [1, 4]',
  },
  {
    key: 'forward',
    label: '② 模型预测',
    title: 'TRANSFORMER / NEXT TOKEN',
    note: 'Embedding、RoPE 自注意力与 SwiGLU 逐层改写 residual stream，LM head 输出完整词表 logits。',
    tokens: [
      { piece: 'girl', id: 1843, width: 58, tone: 'lime' },
      { piece: 'boy', id: 2234, width: 27, tone: 'violet' },
      { piece: 'dragon', id: 6148, width: 11, tone: 'orange' },
      { piece: 'house', id: 3921, width: 4, tone: 'blue' },
    ],
    target: 'logits 形状 [B, T, V] = [1, 4, 10,000]',
  },
  {
    key: 'learn',
    label: '③ 损失与更新',
    title: 'CROSS ENTROPY / ADAMW',
    note: '正确下一个 token 是 “girl”。交叉熵只读取目标位置的 log-prob，AdamW 再更新所有参与计算的参数。',
    tokens: [
      { piece: 'girl · after', id: 1843, width: 73, tone: 'lime' },
      { piece: 'girl · before', id: 1843, width: 38, tone: 'violet' },
      { piece: 'loss', id: 0, width: 31, tone: 'orange' },
      { piece: 'grad norm', id: 0, width: 46, tone: 'blue' },
    ],
    target: '一次 step：forward → loss → backward → clip → AdamW',
  },
] as const;

export function TokenJourneyConsole() {
  const [active, setActive] = useState(0);
  const stage = consoleStages[active];

  return (
    <div className="journey-console">
      <div className="journey-console-chrome">
        <div aria-hidden="true"><i /><i /><i /></div>
        <span>train_step / sample_0007</span>
        <b>LIVE</b>
      </div>
      <div className="journey-console-prompt">
        <span>PROMPT</span>
        <p>Once upon a time there was a little</p>
      </div>
      <div className="journey-console-title">
        <span>{stage.title}</span>
        <code>{stage.target}</code>
      </div>
      <div className="journey-token-list">
        {stage.tokens.map((token) => (
          <div className="journey-token-row" key={`${stage.key}-${token.piece}`}>
            <strong>{token.piece}</strong>
            <div className="journey-token-bar">
              <i data-tone={token.tone} style={{ width: `${token.width}%` }} />
              <em>{stage.key === 'tokenize' ? `id ${token.id}` : `${token.width}%`}</em>
            </div>
          </div>
        ))}
      </div>
      <div className="journey-console-note"><span>↳</span><p>{stage.note}</p></div>
      <div className="journey-console-tabs" role="tablist" aria-label="一次语言模型训练步骤">
        {consoleStages.map((item, index) => (
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

const routes = [
  {
    key: 'first',
    label: '第一次系统学习',
    icon: BookOpen,
    title: '先建立一条不会断的知识链',
    description: '从 Python / 张量 / 概率先修开始，随后把 tokenizer、Transformer、训练与生成串成闭环。',
    steps: ['补齐张量与概率先修', '看懂 byte-level BPE', '手算一次 attention', '训练第一个 TinyStories LM'],
    href: '/docs/roadmap',
  },
  {
    key: 'builder',
    label: '准备完成作业',
    icon: Wrench,
    title: '按测试依赖顺序写出完整系统',
    description: '每一步都给出实现、形状、逐段解读、测试命令和典型失败，不把关键代码留作“自行完成”。',
    steps: ['A1：模型与训练', 'A2：kernel 与并行', 'A3：缩放律实验', 'A4/A5：数据与对齐'],
    href: '/docs/assignments',
  },
  {
    key: 'systems',
    label: '想吃透工程',
    icon: Cpu,
    title: '从 FLOPs 走到真实 GPU 时间',
    description: '沿 profiler、Triton、FlashAttention、DDP/FSDP 与推理 KV cache，理解性能瓶颈为何出现。',
    steps: ['算术强度与 roofline', '写 fused Triton kernel', '拆解 FlashAttention', '分析通信与推理吞吐'],
    href: '/docs/systems/gpu-kernels',
  },
] as const;

export function LearningRouteSelector() {
  const [active, setActive] = useState<(typeof routes)[number]['key']>('first');
  const route = routes.find((item) => item.key === active) ?? routes[0];

  return (
    <div className="journey-route-picker">
      <div className="journey-route-tabs" role="tablist" aria-label="选择学习路线">
        {routes.map(({ key, label, icon: Icon }) => (
          <button key={key} role="tab" aria-selected={key === active} onClick={() => setActive(key)}>
            <Icon size={16} /> {label}
          </button>
        ))}
      </div>
      <div className="journey-route-panel">
        <span>RECOMMENDED ROUTE</span>
        <h3>{route.title}</h3>
        <p>{route.description}</p>
        <ol>
          {route.steps.map((step, index) => <li key={step}><b>{index + 1}</b>{step}</li>)}
        </ol>
        <Link href={route.href}>沿这条路线开始 <ArrowRight size={16} /></Link>
      </div>
    </div>
  );
}

export function CodeProofStrip() {
  return (
    <div className="journey-code-proof" aria-label="课程实现原则">
      <span><Braces size={14} /> 公式紧贴张量形状</span>
      <span><Cpu size={14} /> 代码经过官方测试核对</span>
      <span><BookOpen size={14} /> 讲义逐讲对照</span>
    </div>
  );
}
