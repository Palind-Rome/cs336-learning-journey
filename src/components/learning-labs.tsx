'use client';

import { useMemo, useState } from 'react';
import { ArrowRight, Check, Circle, Info, RotateCcw, X } from 'lucide-react';
import type { ReactNode } from 'react';

/* ──  TensorShapeLab  ── */
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

  return (
    <div className="learning-panel">
      <div className="font-bold text-lg">张量形状实验室</div>
      <p className="mb-5 mt-2 text-sm text-fd-muted-foreground">拖动参数，看多头注意力如何拆分，以及注意力矩阵为何随序列长度平方增长。</p>
      <div className="grid gap-4 sm:grid-cols-2">
        {sliders.map((item) => (
          <label key={item.label} className="text-sm">
            <span className="flex justify-between"><b>{item.label}</b><code>{item.value}</code></span>
            <input className="learning-slider mt-2 w-full" type="range" min={item.min} max={item.max} step={item.step ?? 1} value={item.value} onChange={(e) => item.set(Number(e.target.value))} />
          </label>
        ))}
      </div>
      <div className="mt-5 grid gap-3 rounded-xl bg-fd-muted p-4 text-sm sm:grid-cols-3">
        <div>Q/K/V<br /><code>[{batch}, {heads}, {tokens}, {headDim}]</code></div>
        <div>注意力分数<br /><code>[{batch}, {heads}, {tokens}, {tokens}]</code></div>
        <div>分数元素数<br /><code>{scoreElements.toLocaleString()}</code></div>
      </div>
      {!Number.isInteger(headDim) && <p className="mt-3 text-sm text-red-500">d_model 必须能被头数整除；这组参数无法直接 reshape。</p>}
    </div>
  );
}

/* ──  ScalingLab  ── */
export function ScalingLab() {
  const [params, setParams] = useState(100);
  const [tokens, setTokens] = useState(2);
  const flops = 6 * params * 1e6 * tokens * 1e9;
  const days = flops / (312e12 * 0.4) / 86400;

  return (
    <div className="learning-panel">
      <div className="font-bold text-lg">训练预算估算器</div>
      <p className="mb-5 mt-2 text-sm text-fd-muted-foreground">粗略使用训练 FLOPs ≈ 6ND；硬件示例按单张 A100 312 TFLOP/s、40% MFU 估计。</p>
      <label className="block text-sm">
        <span className="flex justify-between"><b>参数 N</b><code>{params}M</code></span>
        <input className="learning-slider w-full" type="range" min="20" max="2000" step="20" value={params} onChange={(e) => setParams(Number(e.target.value))} />
      </label>
      <label className="mt-4 block text-sm">
        <span className="flex justify-between"><b>训练 token D</b><code>{tokens}B</code></span>
        <input className="learning-slider w-full" type="range" min="1" max="50" value={tokens} onChange={(e) => setTokens(Number(e.target.value))} />
      </label>
      <div className="mt-5 grid grid-cols-2 gap-3">
        <div className="rounded-xl bg-fd-muted p-4"><div className="text-xs text-fd-muted-foreground">总训练计算</div><b>{flops.toExponential(2)} FLOPs</b></div>
        <div className="rounded-xl bg-fd-muted p-4"><div className="text-xs text-fd-muted-foreground">单卡粗估</div><b>{days.toFixed(1)} A100-days</b></div>
      </div>
    </div>
  );
}

/* ──  CheckpointQuiz  ── */
export function CheckpointQuiz({ question, options, answer, explanation }: { question: string; options: string[]; answer: number; explanation: string }) {
  const [picked, setPicked] = useState<number | null>(null);
  const isCorrect = picked === answer;

  return (
    <div className="checkpoint-quiz">
      <div className="checkpoint-quiz-head">
        <div>
          <span>CHECKPOINT</span>
          <h3>理解检查</h3>
        </div>
      </div>
      <div className="checkpoint-quiz-body">
        <p>{question}</p>
        <div className="quiz-options">
          {options.map((option, index) => (
            <button
              key={option}
              onClick={() => setPicked(index)}
              className={`q-pick-btn${picked === index ? ' q-picked' : ''}${picked !== null && index === answer ? ' q-correct' : ''}`}
            >
              {option}
            </button>
          ))}
        </div>
        {picked !== null && (
          <div className="checkpoint-quiz-feedback">
            {isCorrect ? <Check size={15} className="inline text-emerald-600 mr-2" /> : <X size={15} className="inline text-red-500 mr-2" />}
            <b>{isCorrect ? '答对了。' : '再想想。'}</b>{' '}{explanation}
          </div>
        )}
      </div>
    </div>
  );
}

/* ──  AssignmentStepper  ── */
export function AssignmentStepper({ items }: { items: string[] }) {
  const [done, setDone] = useState<boolean[]>(() => items.map(() => false));
  const count = useMemo(() => done.filter(Boolean).length, [done]);

  return (
    <div className="learning-panel">
      <div className="flex items-center justify-between">
        <b>动手进度</b>
        <span className="text-sm text-fd-muted-foreground">{count}/{items.length}</span>
      </div>
      <div className="my-4 h-2 overflow-hidden rounded-full bg-fd-muted">
        <div className="learning-progress-fill" style={{ width: `${(count / items.length) * 100}%` }} />
      </div>
      <div className="grid gap-2">
        {items.map((item, i) => (
          <button
            key={item}
            onClick={() => setDone((v) => v.map((x, j) => (j === i ? !x : x)))}
            className="flex items-start gap-3 rounded-lg p-2 text-left text-sm hover:bg-fd-accent"
          >
            {done[i] ? <Check className="mt-0.5 size-4 text-emerald-600" /> : <Circle className="mt-0.5 size-4 text-fd-muted-foreground" />}
            <span className={done[i] ? 'text-fd-muted-foreground line-through' : ''}>{item}</span>
          </button>
        ))}
      </div>
      {count > 0 && (
        <button onClick={() => setDone(items.map(() => false))} className="mt-3 flex items-center gap-1 text-xs text-fd-muted-foreground">
          <RotateCcw className="size-3" /> 重置
        </button>
      )}
    </div>
  );
}

/* ──  TermDeck  ── */
type TermItem = {
  term: string;
  expansion?: string;
  chinese: string;
  explanation: string;
  example?: string;
};

export function TermDeck({ items, title = '第一次遇见这些词' }: { items: TermItem[]; title?: string }) {
  const [active, setActive] = useState(0);
  const item = items[active] ?? items[0];
  if (!item) return null;

  return (
    <div className="term-deck">
      <div className="lab-heading">
        <div>
          <span>TERM DECK</span>
          <h3>{title}</h3>
        </div>
        <b>{String(active + 1).padStart(2, '0')} / {String(items.length).padStart(2, '0')}</b>
      </div>
      <div className="term-deck-body">
        <div className="term-deck-tabs" role="tablist" aria-label={title}>
          {items.map((candidate, index) => (
            <button
              key={`${candidate.term}-${index}`}
              role="tab"
              aria-selected={index === active}
              onClick={() => setActive(index)}
            >
              {candidate.term}
            </button>
          ))}
        </div>
        <div className="term-deck-panel">
          <span>{item.expansion ?? item.term}</span>
          <h4>{item.term} · {item.chinese}</h4>
          <p>{item.explanation}</p>
          {item.example && <code>{item.example}</code>}
        </div>
      </div>
      <p className="lab-footnote"><Info size={14} /> 点击缩写切换；后文再次出现时，先把它读成这里的完整含义。</p>
    </div>
  );
}

/* ──  ExecutionTrace  ── */
type TraceStep = {
  label: string;
  title: string;
  input: string;
  output: string;
  explanation: string;
};

export function ExecutionTrace({ title, steps }: { title: string; steps: TraceStep[] }) {
  const [active, setActive] = useState(0);
  const step = steps[active] ?? steps[0];
  if (!step) return null;

  return (
    <div className="execution-trace">
      <div className="lab-heading" style={{ flexDirection: 'column', alignItems: 'flex-start' }}>
        <span>EXECUTION TRACE</span>
        <h3>{title}</h3>
      </div>
      <div className="execution-trace-track" role="tablist" aria-label={title}>
        {steps.map((candidate, index) => (
          <button
            key={`${candidate.label}-${index}`}
            role="tab"
            aria-selected={index === active}
            onClick={() => setActive(index)}
          >
            <b>{index + 1}</b>
            <span>{candidate.label}</span>
            {index < steps.length - 1 && <ArrowRight size={13} />}
          </button>
        ))}
      </div>
      <div className="execution-trace-panel">
        <span>STEP {String(active + 1).padStart(2, '0')}</span>
        <h4>{step.title}</h4>
        <div>
          <code>IN&nbsp;&nbsp; {step.input}</code>
          <code>OUT&nbsp; {step.output}</code>
        </div>
        <p>{step.explanation}</p>
      </div>
    </div>
  );
}

/* ──  AdvantageNormalizerLab  ── */
export function AdvantageNormalizerLab() {
  const [groupSize, setGroupSize] = useState(8);
  const [correct, setCorrect] = useState(2);
  const [normalizer, setNormalizer] = useState<'std' | 'none' | 'mean'>('std');
  const boundedCorrect = Math.min(correct, groupSize);
  const mean = boundedCorrect / groupSize;
  const sampleVariance = groupSize > 1
    ? (boundedCorrect * (1 - mean) ** 2 + (groupSize - boundedCorrect) * mean ** 2) / (groupSize - 1)
    : 0;
  const denominator = normalizer === 'std'
    ? Math.sqrt(sampleVariance) + 1e-6
    : normalizer === 'mean'
      ? mean + 1e-6
      : 1;
  const positive = boundedCorrect ? (1 - mean) / denominator : null;
  const negative = boundedCorrect < groupSize ? -mean / denominator : null;

  return (
    <div className="learning-panel">
      <div className="font-bold text-lg">Group advantage 实验台</div>
      <p className="mb-5 mt-2 text-sm text-fd-muted-foreground">reward 为 0/1；std 使用与课程测试一致的样本标准差。拖动后观察难度重加权。</p>
      <div className="grid gap-4 sm:grid-cols-2">
        <label className="text-sm">
          <span className="flex justify-between"><b>group size G</b><code>{groupSize}</code></span>
          <input className="learning-slider mt-2 w-full" type="range" min="2" max="16" value={groupSize} onChange={(event) => { const value = Number(event.target.value); setGroupSize(value); setCorrect((current) => Math.min(current, value)); }} />
        </label>
        <label className="text-sm">
          <span className="flex justify-between"><b>correct rollouts</b><code>{boundedCorrect}</code></span>
          <input className="learning-slider mt-2 w-full" type="range" min="0" max={groupSize} value={boundedCorrect} onChange={(event) => setCorrect(Number(event.target.value))} />
        </label>
      </div>
      <div className="mt-4 flex flex-wrap gap-2">
        {(['std', 'none', 'mean'] as const).map((item) => (
          <button
            key={item}
            onClick={() => setNormalizer(item)}
            className={`rounded-full border px-3 py-1.5 text-sm ${normalizer === item ? 'learning-pill-active' : 'hover:bg-fd-accent'}`}
          >
            {item === 'std' ? 'GRPO · ÷ std' : item === 'mean' ? 'MaxRL · ÷ mean' : 'Dr.GRPO · 不除'}
          </button>
        ))}
      </div>
      <div className="mt-5 grid gap-3 sm:grid-cols-4">
        <div className="rounded-xl bg-fd-muted p-3"><span className="text-xs text-fd-muted-foreground">group mean</span><div className="font-mono font-bold">{mean.toFixed(4)}</div></div>
        <div className="rounded-xl bg-fd-muted p-3"><span className="text-xs text-fd-muted-foreground">sample std</span><div className="font-mono font-bold">{Math.sqrt(sampleVariance).toFixed(4)}</div></div>
        <div className="rounded-xl bg-emerald-500/10 p-3"><span className="text-xs text-fd-muted-foreground">correct advantage</span><div className="font-mono font-bold text-emerald-600">{positive === null ? '—' : positive.toFixed(4)}</div></div>
        <div className="rounded-xl bg-red-500/10 p-3"><span className="text-xs text-fd-muted-foreground">wrong advantage</span><div className="font-mono font-bold text-red-600">{negative === null ? '—' : negative.toFixed(4)}</div></div>
      </div>
      {(boundedCorrect === 0 || boundedCorrect === groupSize) && (
        <p className="mt-3 text-sm text-amber-600">组内 reward 完全相同，减去 group mean 后所有 advantage 都为 0，这组不会提供梯度。</p>
      )}
    </div>
  );
}

/* ──  ImportanceClipLab  ── */
export function ImportanceClipLab() {
  const [logRatio, setLogRatio] = useState(0.25);
  const [advantage, setAdvantage] = useState<1 | -1>(1);
  const [epsilon, setEpsilon] = useState(0.2);
  const ratio = Math.exp(logRatio);
  const clippedRatio = Math.min(1 + epsilon, Math.max(1 - epsilon, ratio));
  const unclipped = advantage * ratio;
  const clipped = advantage * clippedRatio;
  const surrogate = Math.min(unclipped, clipped);
  const gradientActive = advantage > 0 ? ratio <= 1 + epsilon : ratio >= 1 - epsilon;

  return (
    <div className="learning-panel">
      <div className="font-bold text-lg">PPO / GRPO clipping 实验台</div>
      <p className="mb-5 mt-2 text-sm text-fd-muted-foreground">ratio = exp(log πθ − log πold)。目标取 min(A·ratio, A·clip(ratio))。</p>
      <label className="block text-sm">
        <span className="flex justify-between"><b>log importance ratio</b><code>{logRatio.toFixed(2)}</code></span>
        <input className="learning-slider mt-2 w-full" type="range" min="-1.2" max="1.2" step="0.05" value={logRatio} onChange={(event) => setLogRatio(Number(event.target.value))} />
      </label>
      <label className="mt-4 block text-sm">
        <span className="flex justify-between"><b>clip ε</b><code>{epsilon.toFixed(2)}</code></span>
        <input className="learning-slider mt-2 w-full" type="range" min="0.05" max="0.5" step="0.05" value={epsilon} onChange={(event) => setEpsilon(Number(event.target.value))} />
      </label>
      <div className="mt-4 flex gap-2">
        <button onClick={() => setAdvantage(1)} className={`rounded-full border px-3 py-1.5 text-sm ${advantage > 0 ? 'border-emerald-500 bg-emerald-500/10' : ''}`}>A = +1</button>
        <button onClick={() => setAdvantage(-1)} className={`rounded-full border px-3 py-1.5 text-sm ${advantage < 0 ? 'border-red-500 bg-red-500/10' : ''}`}>A = −1</button>
      </div>
      <div className="mt-5 grid gap-3 sm:grid-cols-4">
        <div className="rounded-xl bg-fd-muted p-3"><span className="text-xs text-fd-muted-foreground">ratio</span><div className="font-mono font-bold">{ratio.toFixed(4)}</div></div>
        <div className="rounded-xl bg-fd-muted p-3"><span className="text-xs text-fd-muted-foreground">clipped ratio</span><div className="font-mono font-bold">{clippedRatio.toFixed(4)}</div></div>
        <div className="rounded-xl bg-fd-muted p-3"><span className="text-xs text-fd-muted-foreground">surrogate</span><div className="font-mono font-bold">{surrogate.toFixed(4)}</div></div>
        <div className={`rounded-xl p-3 ${gradientActive ? 'bg-emerald-500/10' : 'bg-amber-500/10'}`}><span className="text-xs text-fd-muted-foreground">该 token 梯度</span><div className="font-bold">{gradientActive ? '继续更新' : '被截断为 0'}</div></div>
      </div>
    </div>
  );
}

/* ──  SourceBrief  ── */
export function SourceBrief({ title, label = '题面译文', children }: { title: string; label?: string; children: ReactNode }) {
  return (
    <section className="source-brief">
      <div className="source-brief-heading"><span>{label}</span><strong>{title}</strong></div>
      <div className="source-brief-body prose">{children}</div>
    </section>
  );
}

/* ──  ConceptBridge  ── */
export function ConceptBridge({ title, label = '理解桥', children }: { title: string; label?: string; children: ReactNode }) {
  return (
    <aside className="concept-bridge">
      <div className="concept-bridge-heading"><span>{label}</span><strong>{title}</strong></div>
      <div className="concept-bridge-body prose">{children}</div>
    </aside>
  );
}

/* ──  CodeWalkthrough  ── */
type CodeWalkthroughStep = {
  lines: string;
  title: string;
  code: string;
  explanation: string;
  invariant?: string;
};

export function CodeWalkthrough({ title, file, steps }: { title: string; file: string; steps: CodeWalkthroughStep[] }) {
  const [active, setActive] = useState(0);
  const step = steps[active] ?? steps[0];
  if (!step) return null;

  return (
    <section className="code-walkthrough">
      <div className="lab-heading">
        <div>
          <span>CODE WALKTHROUGH</span>
          <h3>{title}</h3>
        </div>
        <code>{file}</code>
      </div>
      <div className="code-walkthrough-layout">
        <div className="code-walkthrough-tabs" role="tablist" aria-label={`${title} 代码分段讲解`}>
          {steps.map((candidate, index) => (
            <button
              key={`${candidate.lines}-${candidate.title}`}
              role="tab"
              aria-selected={index === active}
              onClick={() => setActive(index)}
            >
              <span>{candidate.lines}</span>
              <b>{candidate.title}</b>
            </button>
          ))}
        </div>
        <div className="code-walkthrough-panel">
          <span>{step.lines}</span>
          <h4>{step.title}</h4>
          <pre><code>{step.code}</code></pre>
          <p>{step.explanation}</p>
          {step.invariant && <aside><b>检查点</b> {step.invariant}</aside>}
        </div>
      </div>
    </section>
  );
}
