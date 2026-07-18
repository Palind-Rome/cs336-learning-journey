import Link from 'next/link';
import { ArrowRight, BookOpen, Braces, Cpu, Database, GitBranch, Sparkles } from 'lucide-react';

const tracks = [
  { icon: Braces, label: '01 · 基础', title: 'Tokenizer 与 Transformer', text: '亲手把文本变成 token，再把 token 变成下一个词的概率。', href: '/docs/foundations/tokenization' },
  { icon: Cpu, label: '02 · 系统', title: 'GPU、Triton 与并行', text: '算清 FLOPs、显存和通信，让正确的模型真正跑得快。', href: '/docs/systems/gpu-kernels' },
  { icon: GitBranch, label: '03 · 规模', title: 'Scaling Laws', text: '用小实验预测大训练，理解参数、数据、算力之间的权衡。', href: '/docs/scaling/laws' },
  { icon: Database, label: '04 · 数据', title: '清洗、去重与评测', text: '从 Common Crawl 到可训练语料：质量决定模型能力的上限。', href: '/docs/data/pipeline' },
  { icon: Sparkles, label: '05 · 对齐', title: 'SFT、DPO 与 GRPO', text: '让基础模型学会回答、推理，并在偏好与奖励下持续改进。', href: '/docs/alignment/post-training' },
];

export default function HomePage() {
  return (
    <main className="relative overflow-hidden">
      <div className="hero-grid pointer-events-none absolute inset-x-0 top-0 h-[760px]" />
      <section className="relative mx-auto flex min-h-[690px] w-full min-w-0 max-w-[100vw] flex-col items-center justify-center overflow-hidden px-6 pb-16 pt-28 text-center sm:max-w-7xl">
        <div className="mb-7 inline-flex max-w-full items-center gap-2 whitespace-normal rounded-full border bg-fd-card/70 px-4 py-2 text-xs shadow-sm backdrop-blur sm:text-sm">
          <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-400" />
          Stanford CS336 · Spring 2026 中文学习版
        </div>
        <h1 className="w-full max-w-full text-balance text-4xl font-black leading-[1.08] tracking-[-0.045em] sm:max-w-5xl sm:text-7xl">
          不只会调用模型，<br/><span className="gradient-text">从零造出语言模型。</span>
        </h1>
        <p className="mt-7 w-full max-w-2xl break-words text-pretty text-base leading-8 text-fd-muted-foreground sm:text-xl">
          一条面向 LLM 初学者的完整路径：概念先修、算法推导、源码解剖、交互实验，以及五次课程作业的测试驱动带练。
        </p>
        <div className="mt-10 flex max-w-full flex-wrap justify-center gap-3">
          <Link href="/docs/roadmap" className="inline-flex items-center gap-2 rounded-xl bg-fd-primary px-6 py-3 font-semibold text-fd-primary-foreground shadow-lg transition hover:-translate-y-0.5">
            开始学习 <ArrowRight className="size-4" />
          </Link>
          <Link href="/docs/assignments" className="inline-flex items-center gap-2 rounded-xl border bg-fd-background/70 px-6 py-3 font-semibold backdrop-blur transition hover:bg-fd-accent">
            <BookOpen className="size-4" /> 进入作业工坊
          </Link>
        </div>
        <div className="glass-card mt-14 grid w-full min-w-0 max-w-4xl grid-cols-2 overflow-hidden rounded-2xl sm:grid-cols-4">
          {[['17', '讲课程地图'], ['5', '次作业带练'], ['4', '类交互组件'], ['0→1', '完整训练链路']].map(([n, t]) => (
            <div key={t} className="border-r border-fd-border/70 p-5 last:border-r-0">
              <div className="text-2xl font-black">{n}</div><div className="mt-1 text-xs text-fd-muted-foreground">{t}</div>
            </div>
          ))}
        </div>
      </section>

      <section className="mx-auto max-w-7xl px-6 py-20">
        <div className="mb-10 max-w-2xl">
          <p className="mb-3 text-sm font-bold uppercase tracking-[.2em] text-violet-500">Build the whole stack</p>
          <h2 className="text-3xl font-bold tracking-tight sm:text-4xl">沿着一条主线，把碎片知识连成系统</h2>
          <p className="mt-4 leading-7 text-fd-muted-foreground">每一站都回答三个问题：它解决什么、为什么这样设计、如何在作业代码和测试中看见它。</p>
        </div>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-5">
          {tracks.map(({ icon: Icon, ...item }) => (
            <Link key={item.label} href={item.href} className="group rounded-2xl border bg-fd-card/60 p-5 transition hover:-translate-y-1 hover:border-violet-400/60 hover:shadow-xl">
              <Icon className="mb-8 size-6 text-violet-500" />
              <div className="text-xs font-bold text-fd-muted-foreground">{item.label}</div>
              <h3 className="mt-2 font-bold">{item.title}</h3>
              <p className="mt-3 text-sm leading-6 text-fd-muted-foreground">{item.text}</p>
            </Link>
          ))}
        </div>
      </section>

      <section className="mx-auto max-w-7xl px-6 pb-24">
        <div className="overflow-hidden rounded-3xl border bg-[linear-gradient(120deg,rgba(109,93,252,.12),rgba(34,211,238,.08))] p-8 sm:p-12">
          <div className="grid items-center gap-10 lg:grid-cols-[1fr_.8fr]">
            <div><h2 className="text-3xl font-bold">读、算、写、测：四步闭环</h2><p className="mt-4 max-w-xl leading-7 text-fd-muted-foreground">先用直觉建立心智模型，再推一遍公式；接着只实现一个小接口，用官方测试得到反馈；最后回到系统视角解释性能与取舍。</p></div>
            <div className="grid grid-cols-2 gap-3 text-sm">{['① 直觉与术语', '② 公式与形状', '③ 代码骨架', '④ 测试与复盘'].map((x) => <div key={x} className="rounded-xl border bg-fd-background/70 p-4 font-semibold">{x}</div>)}</div>
          </div>
        </div>
      </section>
    </main>
  );
}
