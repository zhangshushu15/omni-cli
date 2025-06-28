/**
 * @license
 * Copyright 2025 The Omni Code Project
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  GenerateContentResponse,
  GenerateContentParameters,
  CountTokensParameters,
  CountTokensResponse,
  EmbedContentParameters,
  EmbedContentResponse,
  Content,
  ContentListUnion,
  FinishReason,
  FunctionDeclaration,
  ToolListUnion,
} from '@google/genai';
import {
  AbstractProvider,
  ProviderAPIError,
  ProviderConfigurationError,
} from './baseProvider.js';
import { LLMProvider, ProviderConfig } from '../config/models.js';

interface AnthropicMessage {
  role: 'user' | 'assistant' | 'system';
  content:
    | string
    | Array<{
        type: 'text' | 'tool_use' | 'tool_result';
        text?: string;
        id?: string;
        name?: string;
        input?: Record<string, unknown>;
        tool_use_id?: string;
        content?: string;
      }>;
}

interface AnthropicTool {
  name: string;
  description: string;
  input_schema: Record<string, unknown>;
}

interface AnthropicChatRequest {
  model: string;
  messages: AnthropicMessage[];
  max_tokens?: number;
  temperature?: number;
  stop_sequences?: string[];
  stream?: boolean;
  tools?: AnthropicTool[];
}

interface AnthropicChatResponse {
  id: string;
  type: 'message';
  role: 'assistant';
  content: Array<{
    type: 'text' | 'tool_use';
    text?: string;
    id?: string;
    name?: string;
    input?: Record<string, unknown>;
  }>;
  model: string;
  stop_reason: 'stop_sequence' | 'max_tokens' | 'end_turn' | 'tool_use';
  stop_sequence?: null | string;
  usage: {
    input_tokens: number;
    output_tokens: number;
    cache_creation_input_tokens?: number;
    cache_read_input_tokens?: number;
  };
}

interface AnthropicStreamChunk {
  type:
    | 'message_start'
    | 'message_delta'
    | 'content_block_start'
    | 'content_block_delta'
    | 'content_block_stop'
    | 'message_stop';
  message?: {
    id: string;
    type: 'message';
    role: 'assistant';
    content: Array<{
      type: 'text' | 'tool_use';
      text?: string;
      id?: string;
      name?: string;
      input?: Record<string, unknown>;
    }>;
    model: string;
    stop_reason:
      | null
      | 'stop_sequence'
      | 'max_tokens'
      | 'end_turn'
      | 'tool_use';
    stop_sequence?: null | string;
    usage: {
      input_tokens: number;
      output_tokens: number;
      cache_creation_input_tokens?: number;
      cache_read_input_tokens?: number;
    };
  };
  delta?: {
    text?: string;
    stop_reason?: 'stop_sequence' | 'max_tokens' | 'end_turn' | 'tool_use';
    type?: 'text_delta' | 'input_json_delta';
    partial_json?: string;
  };
  content_block?: {
    type: 'text' | 'tool_use';
    text?: string;
    id?: string;
    name?: string;
    input?: Record<string, unknown>;
  };
  index?: number;
  usage?: {
    input_tokens: number;
    output_tokens: number;
    cache_creation_input_tokens?: number;
    cache_read_input_tokens?: number;
  };
}

/**
 * Anthropic Claude provider that adapts Claude API to the ContentGenerator interface
 */
export class AnthropicProvider extends AbstractProvider {
  readonly provider = LLMProvider.ANTHROPIC;
  protected baseURL: string;
  protected apiKey: string;
  protected apiVersion: string;
  protected defaultMaxTokens = 4096;

  constructor(config: ProviderConfig) {
    super(config);
    this.baseURL = config.baseURL || 'https://api.anthropic.com';
    this.apiKey = config.apiKey || '';
    this.apiVersion = config.apiVersion || '2023-06-01';
  }

  async initialize(): Promise<void> {
    if (!this.apiKey) {
      throw new ProviderConfigurationError(
        this.provider,
        'API key is required for Anthropic Claude',
      );
    }
  }

  async generateContent(
    request: GenerateContentParameters,
  ): Promise<GenerateContentResponse> {
    try {
      const messages = this.convertGeminiToAnthropicMessages(request.contents);
      const model =
        request.model || this.config.model || 'claude-3-sonnet-20240229';
      const tools = this.convertGeminiToolsToAnthropicTools(
        request.config?.tools,
      );

      const anthropicRequest: AnthropicChatRequest = {
        model,
        messages,
        temperature: request.config?.temperature || 0.7,
        max_tokens: request.config?.maxOutputTokens || this.defaultMaxTokens,
        stop_sequences: request.config?.stopSequences,
        stream: false,
      };

      // Add tools if available
      if (tools && tools.length > 0) {
        anthropicRequest.tools = tools;
      }

      const response = await this.makeAnthropicRequest(
        '/v1/messages',
        anthropicRequest,
        {
          signal: request.config?.abortSignal,
        },
      );

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }

      const data: AnthropicChatResponse = await response.json();
      return this.convertAnthropicToGeminiResponse(data);
    } catch (error) {
      throw new ProviderAPIError(
        this.provider,
        `Generation failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
        error instanceof Error ? error : undefined,
      );
    }
  }

  async generateContentStream(
    request: GenerateContentParameters,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    try {
      const messages = this.convertGeminiToAnthropicMessages(request.contents);
      const model =
        request.model || this.config.model || 'claude-3-sonnet-20240229';
      const tools = this.convertGeminiToolsToAnthropicTools(
        request.config?.tools,
      );

      const anthropicRequest: AnthropicChatRequest = {
        model,
        messages,
        temperature: request.config?.temperature || 0.7,
        max_tokens: request.config?.maxOutputTokens || this.defaultMaxTokens,
        stop_sequences: request.config?.stopSequences,
        stream: true,
      };

      // Add tools if available
      if (tools && tools.length > 0) {
        anthropicRequest.tools = tools;
      }

      const response = await this.makeAnthropicRequest(
        '/v1/messages',
        anthropicRequest,
        {
          signal: request.config?.abortSignal,
        },
      );

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }

      return this.parseAnthropicStream(response);
    } catch (error) {
      throw new ProviderAPIError(
        this.provider,
        `Streaming failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
        error instanceof Error ? error : undefined,
      );
    }
  }

  async countTokens(
    request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    // Anthropic doesn't provide a token counting endpoint
    // Provide a reasonable estimation based on text length
    // Rule of thumb: ~4 characters per token for English text

    let totalText = '';

    // Extract text from contents
    const contentArray = Array.isArray(request.contents)
      ? request.contents
      : [request.contents];
    for (const content of contentArray) {
      if (typeof content === 'string') {
        totalText += content;
      } else if (
        content &&
        typeof content === 'object' &&
        'parts' in content &&
        Array.isArray(content.parts)
      ) {
        for (const part of content.parts) {
          if (
            part &&
            typeof part === 'object' &&
            'text' in part &&
            typeof part.text === 'string'
          ) {
            totalText += part.text;
          }
        }
      }
    }

    // Estimate tokens (conservative estimate: ~4 characters per token)
    const estimatedTokens = Math.ceil(totalText.length / 4);

    return {
      totalTokens: estimatedTokens,
    };
  }

  async embedContent(
    _request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    // Anthropic doesn't provide embeddings directly
    throw new ProviderAPIError(
      this.provider,
      'Embeddings not supported by Anthropic API',
    );
  }

  supportsEmbeddings(): boolean {
    return false; // Anthropic doesn't provide embeddings directly
  }

  supportsStreaming(): boolean {
    return true;
  }

  supportsTools(): boolean {
    return true; // Claude supports function calling/tools
  }

  protected async makeAnthropicRequest(
    endpoint: string,
    body: unknown,
    options: { signal?: AbortSignal } = {},
  ): Promise<Response> {
    const url = `${this.baseURL}${endpoint}`;

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Api-Key': this.apiKey,
        'anthropic-version': this.apiVersion,
      },
      body: JSON.stringify(body),
      signal: options.signal,
    });
    return response;
  }

  protected async *parseAnthropicStream(
    response: Response,
  ): AsyncGenerator<GenerateContentResponse> {
    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('Response body is not readable');
    }

    let finalUsage = undefined;
    let buffer = ''; // Buffer to accumulate partial chunks
    let currentToolUse: {
      id?: string;
      name?: string;
      inputJson?: string;
    } | null = null;

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        // Accumulate chunks in buffer
        buffer += new TextDecoder().decode(value);

        // Split by lines and process complete lines
        const lines = buffer.split('\n');

        // Keep the last (potentially incomplete) line in the buffer
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.trim() || !line.startsWith('data: ')) continue;

          try {
            const jsonStr = line.slice(6).trim();
            if (!jsonStr) continue;

            const data = JSON.parse(jsonStr) as AnthropicStreamChunk;

            // Handle content block start (for tool use)
            if (
              data.type === 'content_block_start' &&
              data.content_block?.type === 'tool_use'
            ) {
              currentToolUse = {
                id: data.content_block.id,
                name: data.content_block.name,
                inputJson: '', // Start with empty string to accumulate partial JSON
              };
            }

            // Handle content block delta events (where actual text content is streamed)
            if (data.type === 'content_block_delta' && data.delta?.text) {
              const deltaText = data.delta.text;

              // Yield only the incremental text, not the accumulated text
              const response = new GenerateContentResponse();
              response.candidates = [
                {
                  content: {
                    parts: [
                      {
                        text: deltaText, // Only the new text chunk
                      },
                    ],
                    role: 'model',
                  },
                  finishReason: FinishReason.STOP,
                  index: 0,
                },
              ];

              yield response;
            }

            // Handle tool use input accumulation via input_json_delta
            if (
              data.type === 'content_block_delta' &&
              currentToolUse &&
              data.delta?.type === 'input_json_delta'
            ) {
              // Accumulate the partial JSON string
              currentToolUse.inputJson += data.delta.partial_json || '';
            }

            // Handle content block stop (complete tool use)
            if (
              data.type === 'content_block_stop' &&
              currentToolUse &&
              currentToolUse.name
            ) {
              let parsedInput = {};
              try {
                // Parse the accumulated JSON string
                if (currentToolUse.inputJson) {
                  parsedInput = JSON.parse(currentToolUse.inputJson);
                }
              } catch (parseError) {
                console.warn(
                  '[DEBUG] Failed to parse tool input JSON:',
                  currentToolUse.inputJson,
                  parseError,
                );
              }

              const response = new GenerateContentResponse();
              response.candidates = [
                {
                  content: {
                    parts: [
                      {
                        functionCall: {
                          id: currentToolUse.id,
                          name: currentToolUse.name,
                          args: parsedInput,
                        },
                      },
                    ],
                    role: 'model',
                  },
                  finishReason: FinishReason.STOP,
                  index: 0,
                },
              ];

              yield response;
              currentToolUse = null;
            }

            // Handle message stop event (final usage stats)
            if (data.type === 'message_stop' && data.usage) {
              finalUsage = data.usage;
            }

            // Handle message delta event (for stop reason)
            if (data.type === 'message_delta' && data.delta?.stop_reason) {
              const response = new GenerateContentResponse();
              response.candidates = [
                {
                  content: {
                    parts: [
                      {
                        text: '', // No additional text, just finish reason
                      },
                    ],
                    role: 'model',
                  },
                  finishReason: this.convertFinishReason(
                    data.delta.stop_reason,
                  ),
                  index: 0,
                },
              ];

              if (finalUsage) {
                response.usageMetadata = {
                  promptTokenCount: finalUsage.input_tokens,
                  candidatesTokenCount: finalUsage.output_tokens,
                  totalTokenCount:
                    finalUsage.input_tokens + finalUsage.output_tokens,
                };
              }

              yield response;
            }
          } catch (parseError) {
            console.warn(
              '[DEBUG] Failed to parse streaming chunk:',
              line,
              parseError,
            );
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  protected convertGeminiToAnthropicMessages(
    contents: ContentListUnion,
  ): AnthropicMessage[] {
    const contentArray = Array.isArray(contents) ? contents : [contents];

    const messages: Array<AnthropicMessage | null> = contentArray.map(
      (content) => {
        if (typeof content === 'string') {
          return {
            role: 'user' as const,
            content,
          };
        }

        if ('role' in content) {
          // Handle assistant messages that may contain function calls
          if (content.role === 'model') {
            const messageContent: Array<{
              type: 'text' | 'tool_use';
              text?: string;
              id?: string;
              name?: string;
              input?: Record<string, unknown>;
            }> = [];

            // Extract text and function calls from parts
            if (content.parts) {
              for (const part of content.parts) {
                if (typeof part === 'string') {
                  messageContent.push({ type: 'text', text: part });
                } else if (part && typeof part === 'object') {
                  if (
                    'text' in part &&
                    typeof part.text === 'string' &&
                    part.text.trim()
                  ) {
                    messageContent.push({ type: 'text', text: part.text });
                  } else if ('functionCall' in part && part.functionCall) {
                    // Use the function call's ID if available, otherwise generate one
                    const toolId =
                      part.functionCall.id ||
                      `tool_${Date.now()}_${Math.random().toString(16).slice(2)}`;
                    messageContent.push({
                      type: 'tool_use',
                      id: toolId,
                      name: part.functionCall.name || '',
                      input: part.functionCall.args || {},
                    });
                  }
                }
              }
            }

            // Only create message if there's actual content
            if (messageContent.length > 0) {
              return {
                role: 'assistant' as const,
                content: messageContent,
              } as AnthropicMessage;
            }
            return null; // Skip empty messages
          }

          // Handle user messages that may contain function responses
          if (content.role === 'user') {
            const messageContent: Array<{
              type: 'text' | 'tool_result';
              text?: string;
              tool_use_id?: string;
              content?: string;
            }> = [];

            // Extract text and function responses from parts
            if (content.parts) {
              for (const part of content.parts) {
                if (typeof part === 'string') {
                  messageContent.push({ type: 'text', text: part });
                } else if (part && typeof part === 'object') {
                  if (
                    'text' in part &&
                    typeof part.text === 'string' &&
                    part.text.trim()
                  ) {
                    messageContent.push({ type: 'text', text: part.text });
                  } else if (
                    'functionResponse' in part &&
                    part.functionResponse
                  ) {
                    // Use the function response's ID to match with the corresponding tool_use
                    const toolUseId =
                      part.functionResponse.id ||
                      part.functionResponse.name ||
                      `tool_${Date.now()}`;
                    messageContent.push({
                      type: 'tool_result',
                      tool_use_id: toolUseId,
                      content: JSON.stringify(
                        part.functionResponse.response || {},
                      ),
                    });
                  }
                }
              }
            }

            // If no structured content, fall back to text extraction
            if (messageContent.length === 0) {
              const extractedText = this.extractTextFromContent(content);
              if (extractedText.trim()) {
                messageContent.push({ type: 'text', text: extractedText });
              }
            }

            // Only create message if there's actual content
            if (messageContent.length > 0) {
              return {
                role: 'user' as const,
                content: messageContent,
              } as AnthropicMessage;
            }
            return null; // Skip empty messages
          }

          // Handle system messages
          const extractedText = this.extractTextFromContent(content);
          if (extractedText.trim()) {
            return {
              role: content.role as 'user' | 'system',
              content: extractedText,
            };
          }
          return null; // Skip empty messages
        }

        // Handle Part objects
        const extractedText = this.extractTextFromContent(content as Content);
        if (extractedText.trim()) {
          return {
            role: 'user' as const,
            content: extractedText,
          };
        }
        return null; // Skip empty messages
      },
    );

    // Filter out null entries and return only valid messages
    return messages.filter(
      (message): message is AnthropicMessage => message !== null,
    );
  }

  protected convertAnthropicToGeminiResponse(
    response: AnthropicChatResponse,
  ): GenerateContentResponse {
    const out = new GenerateContentResponse();
    const parts: Array<{
      text?: string;
      functionCall?: {
        id?: string;
        name: string;
        args: Record<string, unknown>;
      };
    }> = [];

    // Process all content blocks
    for (const content of response.content) {
      if (content.type === 'text' && content.text) {
        parts.push({ text: content.text });
      } else if (content.type === 'tool_use' && content.name && content.input) {
        parts.push({
          functionCall: {
            id: content.id,
            name: content.name,
            args: content.input,
          },
        });
      }
    }

    out.candidates = [
      {
        content: {
          parts,
          role: 'model',
        },
        finishReason: this.convertFinishReason(response.stop_reason),
        index: 0,
      },
    ];
    out.usageMetadata = {
      promptTokenCount: response.usage.input_tokens,
      candidatesTokenCount: response.usage.output_tokens,
      totalTokenCount:
        response.usage.input_tokens + response.usage.output_tokens,
    };
    return out;
  }

  protected convertFinishReason(reason: string): FinishReason {
    switch (reason) {
      case 'stop_sequence':
        return FinishReason.STOP;
      case 'max_tokens':
        return FinishReason.MAX_TOKENS;
      case 'end_turn':
        return FinishReason.STOP;
      case 'tool_use':
        return FinishReason.STOP; // Tool use is considered a successful completion
      default:
        return FinishReason.OTHER;
    }
  }

  protected extractTextFromContent(content: Content): string {
    if (typeof content === 'string') {
      return content;
    }

    if ('text' in content && typeof content.text === 'string') {
      return content.text;
    }

    if ('parts' in content && Array.isArray(content.parts)) {
      const extractedText = content.parts
        .map((part) => {
          if (typeof part === 'string') {
            return part;
          }
          if (
            part &&
            typeof part === 'object' &&
            'text' in part &&
            typeof part.text === 'string'
          ) {
            return part.text;
          }
          return '';
        })
        .filter((text) => text.length > 0)
        .join('');

      return extractedText;
    }

    return '';
  }

  protected convertGeminiToolsToAnthropicTools(
    tools: ToolListUnion | undefined,
  ): AnthropicTool[] | undefined {
    if (!tools || tools.length === 0) {
      return undefined;
    }

    // Extract FunctionDeclarations from ToolListUnion
    const functionDeclarations: FunctionDeclaration[] = [];
    for (const tool of tools) {
      if ('functionDeclarations' in tool && tool.functionDeclarations) {
        functionDeclarations.push(...tool.functionDeclarations);
      }
    }

    if (functionDeclarations.length === 0) {
      return undefined;
    }

    return functionDeclarations.map((tool) => ({
      name: tool.name || '',
      description: tool.description || '',
      input_schema: (tool.parameters as Record<string, unknown>) || {},
    }));
  }
}
