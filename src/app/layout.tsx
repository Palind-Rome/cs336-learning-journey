import { Provider } from '@/components/provider';
import './global.css';

export default function Layout({ children }: LayoutProps<'/'>) {
  return (
    <html lang="zh-CN" suppressHydrationWarning>
      <body className="flex flex-col min-h-screen">
        <Provider>{children}</Provider>
      </body>
    </html>
  );
}

export const metadata = {
  metadataBase: new URL(
    process.env.NEXT_PUBLIC_SITE_URL ?? 'https://palind-rome.github.io/cs336-learning-journey/',
  ),
  title: {
    default: 'CS336 · 从零造语言模型',
    template: '%s | CS336 学习之旅',
  },
  description: '面向初学者的 Stanford CS336 中文交互教程：从 tokenizer、Transformer 到系统、缩放律、数据与对齐。',
};
