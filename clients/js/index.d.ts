export const DEFAULT_URL: string;

export class ClefError extends Error {
  constructor(status: number | null, detail: string, requestId?: string | null, retryAfter?: number | null);
  status: number | null;
  detail: string;
  requestId: string | null;
  /** Seconds from the Retry-After header, if sent. */
  retryAfter: number | null;
}

export type Labels = string[] | Record<string, string>;

export interface ClefClientOptions {
  baseUrl?: string;
  apiKey?: string;
  retries?: number;
  backoffMs?: number;
  timeoutMs?: number;
  fetch?: typeof fetch;
}

export interface ClassifyOptions {
  instructions?: string;
  multiLabel?: boolean;
  threshold?: number;
  images?: string[];
  videos?: string[];
  model?: string;
}

export interface ClassifyManyOptions {
  instructions?: string;
  multiLabel?: boolean;
  threshold?: number;
  model?: string;
  chunkSize?: number;
}

export interface ScoreOptions {
  instructions?: string;
  images?: string[];
  videos?: string[];
  model?: string;
}

export interface Classification {
  /** Single-label: the chosen label. Multi-label: top label of `labels`, or null if none. */
  label: string | null;
  labels: string[];
  confidence: number | null;
  scores: Record<string, number>;
  multiLabel: boolean;
  threshold: number | null;
  requestId: string | null;
  usage: Record<string, unknown>;
  timing: Record<string, unknown>;
  classifier: string | null;
  raw: Record<string, any>;
}

export interface ScoreResult {
  score: number;
  level: string;
  levelIndex: number;
  confidence: number | null;
  distribution: Record<string, number>;
  requestId: string | null;
  usage: Record<string, unknown>;
  timing: Record<string, unknown>;
  classifier: string | null;
  raw: Record<string, any>;
}

export interface ClassifierDefinition {
  kind?: 'classify' | 'score';
  labels?: Labels;
  levels?: string[];
  instructions?: string;
  multiLabel?: boolean;
  threshold?: number;
  description?: string;
}

export class Classifier {
  constructor(client: ClefClient, name: string);
  readonly name: string;
  classify(
    input: unknown,
    opts?: { images?: string[]; videos?: string[]; threshold?: number },
  ): Promise<Classification | ScoreResult>;
  classifyMany(inputs: unknown[], opts?: { chunkSize?: number }): Promise<Array<Classification | ScoreResult>>;
  save(def?: ClassifierDefinition): Promise<Record<string, any>>;
  get(): Promise<Record<string, any>>;
  delete(): Promise<{ deleted: string }>;
}

export class ClefClient {
  constructor(opts?: ClefClientOptions);
  baseUrl: string;
  decide(
    state: unknown,
    questions: Record<string, any>,
    opts?: { images?: string[]; videos?: string[]; model?: string },
  ): Promise<Record<string, any>>;
  batch(records: Array<Record<string, any>>): Promise<Record<string, any>>;
  health(): Promise<Record<string, any>>;
  stats(): Promise<Record<string, any>>;
  classify(input: unknown, labels: Labels, opts?: ClassifyOptions): Promise<Classification>;
  classifyMany(inputs: unknown[], labels: Labels, opts?: ClassifyManyOptions): Promise<Classification[]>;
  score(input: unknown, levels: string[], opts?: ScoreOptions): Promise<ScoreResult>;
  classifier(name: string): Classifier;
  listClassifiers(): Promise<Array<Record<string, any>>>;
  saveClassifier(name: string, def?: ClassifierDefinition): Promise<Record<string, any>>;
  getClassifier(name: string): Promise<Record<string, any>>;
  deleteClassifier(name: string): Promise<{ deleted: string }>;
}

export function parseClassification(body: Record<string, any>, requestId?: string | null): Classification;
export function parseScore(body: Record<string, any>, requestId?: string | null): ScoreResult;
export function bytesToDataUrl(bytes: Uint8Array | ArrayBuffer, mime?: string): string;
