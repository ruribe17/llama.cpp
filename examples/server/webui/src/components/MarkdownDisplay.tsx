import React, { useMemo, useState } from 'react';
import Markdown, { ExtraProps } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHightlight from 'rehype-highlight';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';
import remarkBreaks from 'remark-breaks';
import 'katex/dist/katex.min.css';
import { copyStr } from '../utils/misc';

export default function MarkdownDisplay({ content }: { content: string }) {
  const preprocessedContent = useMemo(
    () => preprocessLaTeX(content),
    [content]
  );
  return (
    <Markdown
      remarkPlugins={[remarkGfm, remarkMath, remarkBreaks]}
      rehypePlugins={[rehypeHightlight, rehypeKatex]}
      components={{
        pre: (props) => <Pre {...props} origContent={preprocessedContent} />,
      }}
    >
      {preprocessedContent}
    </Markdown>
  );
}

const Pre: React.ElementType<
  React.ClassAttributes<HTMLPreElement> &
    React.HTMLAttributes<HTMLPreElement> &
    ExtraProps & { origContent: string }
> = ({ node, origContent, ...props }) => {
  const startOffset = node?.position?.start.offset ?? 0;
  const endOffset = node?.position?.end.offset ?? 0;

  const [copied, setCopied] = useState(false);
  const copiedContent = useMemo(
    () =>
      origContent
        .substring(startOffset, endOffset)
        .replace(/^```[^\n]+\n/g, '')
        .replace(/```$/g, ''),
    [origContent, startOffset, endOffset]
  );

  if (!node?.position) {
    return <pre {...props} />;
  }

  return (
    <div className="relative my-4">
      <div
        className="text-right sticky top-4 mb-2 mr-2 h-0"
        onClick={() => {
          copyStr(copiedContent);
          setCopied(true);
        }}
        onMouseLeave={() => setCopied(false)}
      >
        <button className="badge btn-mini">
          {copied ? 'Copied!' : '📋 Copy'}
        </button>
      </div>
      <pre {...props} />
    </div>
  );
};

/**
 * The part below is copied and adapted from:
 * https://github.com/danny-avila/LibreChat/blob/main/client/src/utils/latex.ts
 * (MIT License)
 */

// Regex to check if the processed content contains any potential LaTeX patterns
const containsLatexRegex =
  /\\\(.*?\\\)|\\\[.*?\\\]|\$.*?\$|\\begin\{equation\}.*?\\end\{equation\}/;

// Regex for inline and block LaTeX expressions
const inlineLatex = new RegExp(/\\\((.+?)\\\)/, 'g');
const blockLatex = new RegExp(/\\\[(.*?[^\\])\\\]/, 'gs');

// Function to restore code blocks
const restoreCodeBlocks = (content: string, codeBlocks: string[]) => {
  return content.replace(
    /<<CODE_BLOCK_(\d+)>>/g,
    (_, index) => codeBlocks[index]
  );
};

// Regex to identify code blocks and inline code
const codeBlockRegex = /(```[\s\S]*?```|`.*?`)/g;

export const processLaTeX = (_content: string) => {
  let content = _content;
  // Temporarily replace code blocks and inline code with placeholders
  const codeBlocks: string[] = [];
  let index = 0;
  content = content.replace(codeBlockRegex, (match) => {
    codeBlocks[index] = match;
    return `<<CODE_BLOCK_${index++}>>`;
  });

  // Escape dollar signs followed by a digit or space and digit
  let processedContent = content.replace(/(\$)(?=\s?\d)/g, '\\$');

  // If no LaTeX patterns are found, restore code blocks and return the processed content
  if (!containsLatexRegex.test(processedContent)) {
    return restoreCodeBlocks(processedContent, codeBlocks);
  }

  // Convert LaTeX expressions to a markdown compatible format
  processedContent = processedContent
    .replace(inlineLatex, (_: string, equation: string) => `$${equation}$`) // Convert inline LaTeX
    .replace(blockLatex, (_: string, equation: string) => `$$${equation}$$`); // Convert block LaTeX

  // Restore code blocks
  return restoreCodeBlocks(processedContent, codeBlocks);
};

/**
 * Preprocesses LaTeX content by replacing delimiters and escaping certain characters.
 *
 * @param content The input string containing LaTeX expressions.
 * @returns The processed string with replaced delimiters and escaped characters.
 */
export function preprocessLaTeX(content: string): string {
  // Step 1: Protect code blocks
  const codeBlocks: string[] = [];
  content = content.replace(/(```[\s\S]*?```|`[^`\n]+`)/g, (_, code) => {
    codeBlocks.push(code);
    return `<<CODE_BLOCK_${codeBlocks.length - 1}>>`;
  });

  // Step 2: Protect existing LaTeX expressions
  const latexExpressions: string[] = [];
  content = content.replace(
    /(\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|\\\(.*?\\\))/g,
    (match) => {
      latexExpressions.push(match);
      return `<<LATEX_${latexExpressions.length - 1}>>`;
    }
  );

  // Step 3: Escape dollar signs that are likely currency indicators
  content = content.replace(/\$(?=\d)/g, '\\$');

  // Step 4: Restore LaTeX expressions
  content = content.replace(
    /<<LATEX_(\d+)>>/g,
    (_, index) => latexExpressions[parseInt(index)]
  );

  // Step 5: Restore code blocks
  content = content.replace(
    /<<CODE_BLOCK_(\d+)>>/g,
    (_, index) => codeBlocks[parseInt(index)]
  );

  // Step 6: Apply additional escaping functions
  content = escapeBrackets(content);
  content = escapeMhchem(content);

  return content;
}

export function escapeBrackets(text: string): string {
  const pattern =
    /(```[\S\s]*?```|`.*?`)|\\\[([\S\s]*?[^\\])\\]|\\\((.*?)\\\)/g;
  return text.replace(
    pattern,
    (
      match: string,
      codeBlock: string | undefined,
      squareBracket: string | undefined,
      roundBracket: string | undefined
    ): string => {
      if (codeBlock != null) {
        return codeBlock;
      } else if (squareBracket != null) {
        return `$$${squareBracket}$$`;
      } else if (roundBracket != null) {
        return `$${roundBracket}$`;
      }
      return match;
    }
  );
}

export function escapeMhchem(text: string) {
  return text.replaceAll('$\\ce{', '$\\\\ce{').replaceAll('$\\pu{', '$\\\\pu{');
}
