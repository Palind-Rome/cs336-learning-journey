# CS336 · 从零造语言模型

面向 LLM 初学者的 Stanford CS336 Spring 2026 中文交互学习网站。内容结合课程 17 讲、5 次作业题面、测试与源码，覆盖 tokenizer、Transformer、GPU/分布式、缩放律、数据工程、评测和后训练。

> 本项目是独立学习教程，不是 Stanford 官方翻译或作业答案仓库。作业内容采用接口契约、推导、测试驱动提示和实验方法，引导读者自行实现。

## 本地运行

```bash
npm install
npm run dev
```

打开 <http://localhost:3000>。静态构建：

```bash
npm run build
```

## 内容结构

- `content/docs/lectures/`：17 讲精读，按主题分为五组。
- `content/docs/assignments/`：五次作业测试驱动工坊。
- `content/docs/foundations|systems|scaling|data|alignment/`：按知识依赖组织的主教程。
- `src/components/learning-labs.tsx`：张量形状、预算估算、理解检查和进度组件。

## 部署

构建会静态导出到 `out/`。`.github/workflows/deploy.yml` 在 main 分支更新时构建并部署 GitHub Pages；仓库名会自动作为 `basePath`。

课程原始材料：[CS336 官网](https://cs336.stanford.edu/) · [Stanford CS336 GitHub](https://github.com/stanford-cs336) · [Fumadocs](https://fumadocs.dev)
