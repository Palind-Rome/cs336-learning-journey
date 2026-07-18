import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';
import { appName, gitConfig } from './shared';

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      title: <span className="font-semibold tracking-tight"><span className="hidden sm:inline">{appName}</span><span className="sm:hidden">CS336 学习之旅</span></span>,
      transparentMode: 'top',
    },
    links: [
      { text: '学习路线', url: '/docs/roadmap' },
      { text: '作业工坊', url: '/docs/assignments' },
      { text: '课程官网', url: 'https://cs336.stanford.edu/', external: true },
    ],
    githubUrl: `https://github.com/${gitConfig.user}/${gitConfig.repo}`,
  };
}
