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
  GitBranch,
  Network,
  Sparkles,
} from 'lucide-react';
import { CodeProofStrip, LearningRouteSelector, TokenJourneyConsole } from '@/components/home-interactions';

const stack = [
  {
    number: '01',
    title: '文本变成 token',
    text: '从 Unicode 与 UTF-8 开始，训练 byte-level BPE，并亲手实现可流式编码的 tokenizer。',
    icon: Braces,
    href: '/docs/foundations/tokenization',
    tone: 'lime',
  },
  {
    number: '02',
    title: 'token 变成预测',
    text: '逐层实现 RMSNorm、RoPE、attention 与 SwiGLU，追踪每个张量的形状和数值。',
    icon: Network,
    href: '/docs/foundations/transformer',
    tone: 'violet',
  },
  {
    number: '03',
    title: '预测变成模型',
    text: '把 loss、AdamW、学习率、数据批次与 checkpoint 接成一条可恢复的训练循环。',
    icon: Gauge,
    href: '/docs/foundations/training',
    tone: 'orange',
  },
  {
    number: '04',
    title: '模型跑上集群',
    text: '用 Triton、FlashAttention、DDP/FSDP 与推理系统把理论 FLOPs 变成真实吞吐。',
    icon: Cpu,
    href: '/docs/systems/gpu-kernels',
    tone: 'blue',
  },
  {
    number: '05',
    title: '能力来自全栈',
    text: '用 scaling laws 决定预算，以数据工程塑造分布，再通过 SFT 与 RL 完成后训练。',
    icon: Sparkles,
    href: '/docs/alignment/post-training',
    tone: 'pink',
  },
];

const assignments = [
  { id: 'A1', title: 'Basics', result: '一个从零训练的 Transformer LM', icon: Braces, href: '/docs/assignments/assignment-1' },
  { id: 'A2', title: 'Systems', result: '更快的 kernel、attention 与分布式训练', icon: Cpu, href: '/docs/assignments/assignment-2' },
  { id: 'A3', title: 'Scaling', result: '用小实验预测大模型最优配置', icon: GitBranch, href: '/docs/assignments/assignment-3' },
  { id: 'A4', title: 'Data', result: '可审计的 Common Crawl 清洗管线', icon: Database, href: '/docs/assignments/assignment-4' },
  { id: 'A5', title: 'Alignment', result: 'SFT、RFT 与 GRPO 推理训练闭环', icon: FlaskConical, href: '/docs/assignments/assignment-5' },
];

const sources = ['17 LECTURES', 'A1 · BASICS', 'A2 · SYSTEMS', 'A3 · SCALING', 'A4 · DATA', 'A5 · ALIGNMENT'];

export default function HomePage() {
  return (
    <main className="journey-home">
      <section className="journey-hero">
        <div className="journey-glow journey-glow-one" />
        <div className="journey-glow journey-glow-two" />
        <div className="journey-shell journey-hero-grid">
          <div className="journey-hero-copy">
            <div className="journey-eyebrow"><Sparkles size={14} /> STANFORD CS336 · SPRING 2026 · INTERACTIVE EDITION</div>
            <h1>
              从一个字节开始，
              <span>亲手训练语言模型。</span>
            </h1>
            <p className="journey-hero-lede">
              不把 Transformer 当黑盒。沿着 tokenizer、架构、GPU 系统、缩放律、数据与对齐，读懂公式，也写出真正能通过测试、跑起实验的代码。
            </p>
            <div className="journey-hero-actions">
              <Link className="journey-button journey-button-primary" href="/docs/roadmap">
                开始学习 <ArrowRight size={17} />
              </Link>
              <Link className="journey-button journey-button-ghost" href="/docs/assignments">
                <BookOpen size={17} /> 打开作业工坊
              </Link>
            </div>
            <CodeProofStrip />
          </div>
          <TokenJourneyConsole />
        </div>
        <div className="journey-source-ribbon" aria-label="课程材料覆盖">
          <div className="journey-shell journey-ribbon-track">
            <span className="journey-ribbon-label"><CircleDot size={14} /> COURSE MATERIALS</span>
            {sources.map((source) => <span key={source}>{source}</span>)}
          </div>
        </div>
      </section>

      <section className="journey-section journey-stack-section">
        <div className="journey-shell journey-split-heading">
          <div>
            <p className="journey-section-kicker">ONE END-TO-END STORY</p>
            <h2>不是五门散课，<br />而是一条训练链路。</h2>
          </div>
          <p>
            CS336 的每一部分都在回答同一个问题：怎样从原始网页得到一个会生成、能扩展、可对齐的语言模型。先把因果链连起来，再深入每个算法与实现。
          </p>
        </div>
        <div className="journey-shell journey-stack-flow">
          {stack.map(({ icon: Icon, ...item }, index) => (
            <Link key={item.number} href={item.href} className="journey-stack-card" data-tone={item.tone}>
              <div className="journey-stack-top"><span>{item.number}</span><Icon size={20} /></div>
              <h3>{item.title}</h3>
              <p>{item.text}</p>
              <strong>进入这一层 <ArrowRight size={14} /></strong>
              {index < stack.length - 1 && <i className="journey-stack-link" aria-hidden="true" />}
            </Link>
          ))}
        </div>
      </section>

      <section className="journey-section journey-route-section">
        <div className="journey-shell journey-route-layout">
          <div className="journey-route-intro">
            <p className="journey-section-kicker">CHOOSE YOUR ENTRY POINT</p>
            <h2>从你现在卡住的地方出发。</h2>
            <p>不必按目录硬啃。先选当前目标，路线会告诉你先读什么、接着写什么、用什么测试确认理解。</p>
          </div>
          <LearningRouteSelector />
        </div>
      </section>

      <section className="journey-section journey-assignment-section">
        <div className="journey-shell">
          <div className="journey-section-heading">
            <div>
              <p className="journey-section-kicker">BUILD, RUN, EXPLAIN</p>
              <h2>五次作业，五件能运行的作品。</h2>
            </div>
            <p>每章从接口契约开始，给出完整实现与代码解读，再用官方测试、最小实验和排错路径闭环。</p>
          </div>
          <div className="journey-assignment-grid">
            {assignments.map(({ icon: Icon, ...item }) => (
              <Link key={item.id} href={item.href} className="journey-assignment-card">
                <div><span>{item.id}</span><Icon size={19} /></div>
                <small>{item.title}</small>
                <h3>{item.result}</h3>
                <strong>进入工坊 <ArrowRight size={14} /></strong>
              </Link>
            ))}
          </div>
        </div>
      </section>

      <section className="journey-section journey-evidence-section">
        <div className="journey-shell journey-evidence-card">
          <div><span className="journey-evidence-stamp">READ<br />BUILD<br />VERIFY</span></div>
          <div>
            <p className="journey-section-kicker">THE LEARNING CONTRACT</p>
            <h2>概念要能推导，代码要能运行，结论要能复现。</h2>
            <p>每个陌生缩写先解释，每段关键代码说明输入、输出和不变量；论文结论、课程要求与工程经验分开标注，并回到源码和测试核对。</p>
            <div className="journey-evidence-points">
              <span><Check size={14} /> 术语首次出现即解释</span>
              <span><Check size={14} /> 公式配张量形状</span>
              <span><Check size={14} /> 实现配测试与排错</span>
            </div>
          </div>
          <Link className="journey-button journey-button-light" href="/docs/resources/coverage">
            查看课程地图 <ArrowRight size={17} />
          </Link>
        </div>
      </section>

      <footer className="journey-footer">
        <div className="journey-shell">
          <span>CS336 Learning Journey</span>
          <p>Tokenize the text. Build the model. Scale the system.</p>
          <Link href="/docs">打开完整文档 <ArrowRight size={14} /></Link>
        </div>
      </footer>
    </main>
  );
}
