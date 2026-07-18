'use client';

import { useMemo, useState } from 'react';
import { Check, Circle, RotateCcw } from 'lucide-react';

export function TensorShapeLab() {
  const [batch, setBatch] = useState(2);
  const [tokens, setTokens] = useState(8);
  const [model, setModel] = useState(512);
  const [heads, setHeads] = useState(8);
  const headDim = model / heads;
  const scoreElements = batch * heads * tokens * tokens;
  const sliders: Array<{ label: string; value: number; set: (value: number) => void; min: number; max: number; step?: number }> = [
    { label: '批大小 B', value: batch, set: setBatch, min: 1, max: 16 },
    { label: '序列长度 T', value: tokens, set: setTokens, min: 4, max: 128 },
    { label: '模型维度 d_model', value: model, set: setModel, min: 128, max: 1024, step: 128 },
    { label: '头数 h', value: heads, set: setHeads, min: 1, max: 16 },
  ];
  return <div className="not-prose my-6 rounded-2xl border bg-fd-card p-5">
    <div className="mb-1 font-bold">张量形状实验室</div><p className="mb-5 text-sm text-fd-muted-foreground">拖动参数，看多头注意力如何拆分，以及注意力矩阵为何随序列长度平方增长。</p>
    <div className="grid gap-4 sm:grid-cols-2">{sliders.map((item) => <label key={item.label} className="text-sm"><span className="flex justify-between"><b>{item.label}</b><code>{item.value}</code></span><input className="learning-slider mt-2 w-full" type="range" min={item.min} max={item.max} step={item.step ?? 1} value={item.value} onChange={(e) => item.set(Number(e.target.value))}/></label>)}</div>
    <div className="mt-5 grid gap-3 rounded-xl bg-fd-muted p-4 text-sm sm:grid-cols-3"><div>Q/K/V<br/><code>[{batch}, {heads}, {tokens}, {headDim}]</code></div><div>注意力分数<br/><code>[{batch}, {heads}, {tokens}, {tokens}]</code></div><div>分数元素数<br/><code>{scoreElements.toLocaleString()}</code></div></div>
    {!Number.isInteger(headDim) && <p className="mt-3 text-sm text-red-500">d_model 必须能被头数整除；这组参数无法直接 reshape。</p>}
  </div>;
}

export function ScalingLab() {
  const [params, setParams] = useState(100);
  const [tokens, setTokens] = useState(2);
  const flops = 6 * params * 1e6 * tokens * 1e9;
  const days = flops / (312e12 * .4) / 86400;
  return <div className="not-prose my-6 rounded-2xl border bg-fd-card p-5">
    <div className="font-bold">训练预算估算器</div><p className="mb-5 text-sm text-fd-muted-foreground">粗略使用训练 FLOPs ≈ 6ND；硬件示例按单张 A100 312 TFLOP/s、40% MFU 估计。</p>
    <label className="block text-sm"><span className="flex justify-between"><b>参数 N</b><code>{params}M</code></span><input className="learning-slider w-full" type="range" min="20" max="2000" step="20" value={params} onChange={e=>setParams(Number(e.target.value))}/></label>
    <label className="mt-4 block text-sm"><span className="flex justify-between"><b>训练 token D</b><code>{tokens}B</code></span><input className="learning-slider w-full" type="range" min="1" max="50" value={tokens} onChange={e=>setTokens(Number(e.target.value))}/></label>
    <div className="mt-5 grid grid-cols-2 gap-3"><div className="rounded-xl bg-fd-muted p-4"><div className="text-xs text-fd-muted-foreground">总训练计算</div><b>{flops.toExponential(2)} FLOPs</b></div><div className="rounded-xl bg-fd-muted p-4"><div className="text-xs text-fd-muted-foreground">单卡粗估</div><b>{days.toFixed(1)} A100-days</b></div></div>
  </div>;
}

export function CheckpointQuiz({ question, options, answer, explanation }: { question: string; options: string[]; answer: number; explanation: string }) {
  const [picked, setPicked] = useState<number | null>(null);
  return <div className="not-prose my-6 rounded-2xl border border-violet-400/30 bg-violet-500/5 p-5"><div className="text-xs font-bold uppercase tracking-wider text-violet-500">理解检查</div><p className="my-3 font-semibold">{question}</p><div className="grid gap-2">{options.map((option, index)=><button key={option} onClick={()=>setPicked(index)} className={`rounded-lg border px-4 py-3 text-left text-sm transition ${picked===index ? (index===answer?'border-emerald-500 bg-emerald-500/10':'border-red-500 bg-red-500/10'):'hover:bg-fd-accent'}`}>{option}</button>)}</div>{picked!==null&&<p className="mt-4 text-sm"><b>{picked===answer?'答对了。':'再想一步。'}</b> {explanation}</p>}</div>;
}

export function AssignmentStepper({ items }: { items: string[] }) {
  const [done, setDone] = useState<boolean[]>(()=>items.map(()=>false));
  const count = useMemo(()=>done.filter(Boolean).length,[done]);
  return <div className="not-prose my-6 rounded-2xl border bg-fd-card p-5"><div className="flex items-center justify-between"><b>动手进度</b><span className="text-sm text-fd-muted-foreground">{count}/{items.length}</span></div><div className="my-4 h-2 overflow-hidden rounded-full bg-fd-muted"><div className="h-full bg-gradient-to-r from-violet-500 to-cyan-400 transition-all" style={{width:`${count/items.length*100}%`}}/></div><div className="grid gap-2">{items.map((item,i)=><button key={item} onClick={()=>setDone(v=>v.map((x,j)=>j===i?!x:x))} className="flex items-start gap-3 rounded-lg p-2 text-left text-sm hover:bg-fd-accent">{done[i]?<Check className="mt-0.5 size-4 text-emerald-500"/>:<Circle className="mt-0.5 size-4 text-fd-muted-foreground"/>}<span className={done[i]?'text-fd-muted-foreground line-through':''}>{item}</span></button>)}</div>{count>0&&<button onClick={()=>setDone(items.map(()=>false))} className="mt-3 flex items-center gap-1 text-xs text-fd-muted-foreground"><RotateCcw className="size-3"/>重置</button>}</div>;
}
