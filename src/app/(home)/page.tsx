import Link from 'next/link';
import {
  ArrowRight,
  BookOpen,
  Braces,
  Check,
  CircleDot,
  Cpu,
  Database,
  FlaskConical,
  Gauge,
  Sparkles,
} from 'lucide-react';
import { HeroConsole, PathSelector, ProofLedger } from '@/components/home-interactions';

const stages = [
  {
    number: '01',
    glyph: 'B,T',
    title: '从 bytes 到 logits',
    text: 'Unicode 字符串经过 UTF-8、BPE、Embedding、Transformer，最终变成词表上的概率分布。闭合这条因果链，后续所有优化都有了一个可感知的基点。',
    href: '/docs/foundations',
    bg: 'var(--lab-indigo)',
  },
  {
    number: '02',
    glyph: '⇅',
    title: '让 GPU 不再等待',
    text: '把数学运算映射到 HBM、SRAM 与 Tensor Core；用 tiling、fusion 和 collectives 让同一模型跑出数倍吞吐。',
    href: '/docs/systems',
    bg: 'var(--lab-teal)',
    color: '#11302b',
  },
  {
    number: '03',
    glyph: 'ƒ',
    title: '用小实验预测大训练',
    text: '在受限预算下系统改变 N、D 与 recipe，测量 loss，再外推更大规模的最优配置。这不只是曲线拟合，而是一种决策思维。',
    href: '/docs/scaling',
    bg: 'var(--lab-coral)',
    color: '#3d1b14',
  },
];

const chapters = [
  {
    number: '01',
    title: '基础与架构',
    text: 'Tokenizer、Transformer、AdamW、训练循环——完成语言模型的最小闭环。',
    icon: Braces,
    href: '/docs/foundations',
    tone: 'indigo',
  },
  {
    number: '02',
    title: '系统与效率',
    text: 'FlashAttention、Triton kernel、DDP、FSDP——把数学结果高效映射到 GPU 集群。',
    icon: Cpu,
    href: '/docs/systems',
    tone: 'teal',
  },
  {
    number: '03',
    title: '缩放与推理',
    text: 'IsoFLOPs、Chinchilla、μP、量化与 speculative decoding——用科学方法做大规模决策。',
    icon: Gauge,
    href: '/docs/scaling',
    tone: 'amber',
  },
  {
    number: '04',
    title: '评测与数据',
    text: '从 Common Crawl 到可训练语料：抽取、过滤、去重、混合与审计。',
    icon: Database,
    href: '/docs/data',
    tone: 'coral',
  },
  {
    number: '05',
    title: '对齐与多模态',
    text: 'SFT、DPO、GRPO、RLVR——把续写模型塑造成可控、可评测的助手。',
    icon: FlaskConical,
    href: '/docs/alignment',
    tone: 'indigo',
  },
];

const sources = [
  'CS336 · Stanford 2026',
  '17 Lectures',
  '5 Assignments',
  '9 Executable Notebooks',
  'Triton · FSDP · GRPO',
  'Source-First',
];

const assignments = [
  { id: 'A1', title: 'Basics', result: '从 bytes 训练 Transformer LM', tests: '46 tests passed', href: '/docs/assignments/assignment-1' },
  { id: 'A2', title: 'Systems', result: '写快 attention，分到多 GPU', tests: '10 CPU tests passed', href: '/docs/assignments/assignment-2' },
  { id: 'A3', title: 'Scaling', result: '从 IsoFLOPs 实验预测最优配置', tests: 'checks passed', href: '/docs/assignments/assignment-3' },
  { id: 'A4', title: 'Data', result: '把 Common Crawl 变成训练数据', tests: '18 tests passed', href: '/docs/assignments/assignment-4' },
  { id: 'A5', title: 'Alignment', result: 'GRPO、GSPO、SFT 与 DPO', tests: '26 tests passed', href: '/docs/assignments/assignment-5' },
];

export default function HomePage() {
  return (
    <main className="lab-home">
      {/* ── Hero ── */}
      <section className="home-hero">
        <div className="hero-glow hero-glow-one" />
        <div className="hero-glow hero-glow-two" />
        <div className="home-shell hero-grid">
          <div className="hero-copy">
            <div className="eyebrow">
              <Sparkles size={14} /> LANGUAGE MODELING FROM SCRATCH · 2026
            </div>
            <h1>
              把语言模型拆开，
              <span>再亲手装回去。</span>
            </h1>
            <p className="hero-lede">
              从 UTF-8 的一个 byte 出发，沿着张量、GPU、集群、数据与奖励一路向前。
              每个概念先建立直觉，再推公式、读源码、跑测试，最后变成一件真正能工作的系统。
            </p>
            <div className="hero-actions">
              <Link className="button button-primary" href="/docs/roadmap">
                开始学习 <ArrowRight size={17} />
              </Link>
              <Link className="button button-ghost" href="/docs/assignments">
                <BookOpen size={17} /> 进入作业工坊
              </Link>
            </div>
            <ProofLedger />
          </div>
          <HeroConsole />
        </div>
        <div className="source-ribbon" aria-label="课程概况">
          <div className="home-shell ribbon-track">
            <span className="ribbon-label"><CircleDot size={14} /> CS336 FULL STACK</span>
            {sources.map((source) => <span key={source}>{source}</span>)}
          </div>
        </div>
      </section>

      {/* ── Three stages flow ── */}
      <section className="home-section">
        <div className="home-shell split-heading">
          <div>
            <p className="section-kicker">THREE STAGES OF UNDERSTANDING</p>
            <h2>不只是读懂，<br />是能复现和解释。</h2>
          </div>
          <p>
            每一站都先问「它在解决什么问题」，再用手算一次形状和梯度，最后用官方测试验证你的实现。
            右边的三个步骤是整门课反复使用的认知循环。
          </p>
        </div>
        <div className="home-shell stage-flow">
          {stages.map(({ glyph, title, text, bg, color }, i) => [
            <article key={title}>
              <span>{glyph}</span>
              <div className="flow-symbol" style={{ background: bg, color: color ?? '#f0f0ff' }}>{glyph}</div>
              <h3>{title}</h3>
              <p>{text}</p>
            </article>,
            i < stages.length - 1 && <i key={`arrow-${i}`} aria-hidden="true"><ArrowRight /></i>,
          ])}
        </div>
      </section>

      {/* ── Chapter grid ── */}
      <section className="home-section chapter-section">
        <div className="home-shell">
          <div className="section-heading centered">
            <p className="section-kicker">A COURSE IN DEPENDENCY ORDER</p>
            <h2>五个阶段，一条因果链。</h2>
            <p>每个阶段先建立最小心智模型，再进入对应讲义、源码与作业。后面的知识始终能指回前面的不变量。</p>
          </div>
          <div className="chapter-grid">
            {chapters.map(({ icon: Icon, ...chapter }) => (
              <Link key={chapter.number} href={chapter.href} className="chapter-card" data-tone={chapter.tone}>
                <div className="chapter-top"><span>{chapter.number}</span><Icon size={21} /></div>
                <h3>{chapter.title}</h3>
                <p>{chapter.text}</p>
                <strong>进入章节 <ArrowRight size={15} /></strong>
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* ── Path selector ── */}
      <section className="home-section">
        <div className="home-shell path-layout">
          <div className="path-intro">
            <p className="section-kicker">CHOOSE YOUR ROUTE</p>
            <h2>你不需要按目录硬啃。</h2>
            <p>告诉我你现在更像哪一种学习者，得到一条有明确终点和检验方式的阅读顺序。随时可以切换。</p>
          </div>
          <PathSelector />
        </div>
      </section>

      {/* ── Assignment workbench ── */}
      <section className="assignment-section">
        <div className="home-shell">
          <div className="split-heading" style={{ marginBottom: 46 }}>
            <div>
              <p className="section-kicker">ASSIGNMENT WORKBENCH</p>
              <h2>读完一道题，<br />就把它做成。</h2>
            </div>
            <p>题面译文、必要先修、完整实现、逐段解释与测试结果按同一顺序出现，不需要在答案和题目之间来回翻找。</p>
          </div>
          <div className="assignment-ledger">
            <div className="assignment-header" aria-hidden="true">
              <span>LAB</span><span>TOPIC</span><span>DELIVERABLE</span><span>VERIFICATION</span><span />
            </div>
            {assignments.map((item) => (
              <Link key={item.id} href={item.href} className="assignment-row">
                <b>{item.id}</b>
                <span>{item.title}</span>
                <strong>{item.result}</strong>
                <em><Check size={13} /> {item.tests}</em>
                <ArrowRight size={18} />
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* ── Method card ── */}
      <section className="evidence-section">
        <div className="home-shell evidence-card">
          <div>
            <span className="evidence-stamp">READ<br />TRACE<br />TEST</span>
          </div>
          <div>
            <p className="section-kicker">A REPEATABLE METHOD</p>
            <h2>公式旁边就是 shape，实现通过课程测试，题面与讲义逐段对照。</h2>
            <p>
              遇到陌生算法时，先确认输入、输出、shape 与不变量；接着跟一次最小样例的执行路径；
              最后用测试区分「看起来合理」和「实现正确」。
            </p>
          </div>
          <Link className="button button-light" href="/docs/foundations">
            补齐必要先修 <ArrowRight size={17} />
          </Link>
        </div>
      </section>

      {/* ── Footer ── */}
      <footer className="home-footer">
        <div className="home-shell">
          <span>CS336 / Language Modeling from Scratch</span>
          <p>Read the lecture. Trace the source. Make the test pass.</p>
          <Link href="/docs">打开文档目录 <ArrowRight size={14} /></Link>
        </div>
      </footer>
    </main>
  );
}
