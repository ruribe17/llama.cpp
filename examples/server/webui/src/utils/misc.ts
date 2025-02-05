// @ts-expect-error this package does not have typing
import TextLineStream from 'textlinestream';
import { APIMessage, Message } from './types';

// ponyfill for missing ReadableStream asyncIterator on Safari
import { asyncIterator } from '@sec-ant/readable-stream/ponyfill/asyncIterator';
import { isDev } from '../Config';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const isString = (x: any) => !!x.toLowerCase;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const isBoolean = (x: any) => x === true || x === false;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const isNumeric = (n: any) => !isString(n) && !isNaN(n) && !isBoolean(n);
export const escapeAttr = (str: string) =>
  str.replace(/>/g, '&gt;').replace(/"/g, '&quot;');

// wrapper for SSE
export async function* getSSEStreamAsync(fetchResponse: Response) {
  if (!fetchResponse.body) throw new Error('Response body is empty');
  const lines: ReadableStream<string> = fetchResponse.body
    .pipeThrough(new TextDecoderStream())
    .pipeThrough(new TextLineStream());
  // @ts-expect-error asyncIterator complains about type, but it should work
  for await (const line of asyncIterator(lines)) {
    if (isDev) console.log({ line });
    if (line.startsWith('data:') && !line.endsWith('[DONE]')) {
      const data = JSON.parse(line.slice(5));
      yield data;
    } else if (line.startsWith('error:')) {
      const data = JSON.parse(line.slice(6));
      throw new Error(data.message || 'Unknown error');
    }
  }
}

// copy text to clipboard
export const copyStr = (textToCopy: string) => {
  // Navigator clipboard api needs a secure context (https)
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(textToCopy);
  } else {
    // Use the 'out of viewport hidden text area' trick
    const textArea = document.createElement('textarea');
    textArea.value = textToCopy;
    // Move textarea out of the viewport so it's not visible
    textArea.style.position = 'absolute';
    textArea.style.left = '-999999px';
    document.body.prepend(textArea);
    textArea.select();
    document.execCommand('copy');
  }
};

/**
 * filter out redundant fields upon sending to API
 */
export function normalizeMsgsForAPI(messages: Message[]) {
  return messages.map((msg) => {
    return {
      role: msg.role,
      content: msg.content,
    };
  }) as APIMessage[];
}

/**
 * recommended for DeepsSeek-R1, filter out content between <think> and </think> tags
 */
export function filterThoughtFromMsgs(messages: APIMessage[]) {
  return messages.map((msg) => {
    return {
      role: msg.role,
      content:
        msg.role === 'assistant'
          ? msg.content.split('</think>').at(-1)!.trim()
          : msg.content,
    } as APIMessage;
  });
}

export function classNames(classes: Record<string, boolean>): string {
  return (
    Object.entries(classes)
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      .filter(([_, value]) => value)
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      .map(([key, _]) => key)
      .join(' ')
  );
}
