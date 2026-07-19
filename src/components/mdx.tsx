import defaultMdxComponents from 'fumadocs-ui/mdx';
import * as TabsComponents from 'fumadocs-ui/components/tabs';
import * as AccordionComponents from 'fumadocs-ui/components/accordion';
import { AdvantageNormalizerLab, AssignmentStepper, CheckpointQuiz, CodeWalkthrough, ConceptBridge, ExecutionTrace, ImportanceClipLab, KnowledgeBridge, ScalingLab, SourceBrief, TensorShapeLab, TermDeck } from './learning-labs';
import type { MDXComponents } from 'mdx/types';

export function getMDXComponents(components?: MDXComponents) {
  return {
    ...defaultMdxComponents,
    ...TabsComponents,
    ...AccordionComponents,
    AssignmentStepper,
    AdvantageNormalizerLab,
    CheckpointQuiz,
    CodeWalkthrough,
    ConceptBridge,
    ExecutionTrace,
    ImportanceClipLab,
    KnowledgeBridge,
    ScalingLab,
    SourceBrief,
    TensorShapeLab,
    TermDeck,
    ...components,
  } satisfies MDXComponents;
}

export const useMDXComponents = getMDXComponents;

declare global {
  type MDXProvidedComponents = ReturnType<typeof getMDXComponents>;
}
