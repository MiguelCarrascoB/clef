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

export type JobStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled' | 'interrupted';
export type JobEvent = 'job.succeeded' | 'job.failed' | 'job.cancelled' | 'job.progress';

export interface WebhookSpec {
  url: string;
  /** Signs deliveries: X-Clef-Signature = sha256=HMAC(secret, `${timestamp}.${body}`). */
  secret?: string;
  /** Default: job.succeeded, job.failed, job.cancelled. */
  events?: JobEvent[];
}

export interface JobOptions {
  webhook?: string | WebhookSpec;
  metadata?: Record<string, unknown>;
}

export interface ClassifyJobOptions extends JobOptions {
  /** Name of a saved classify classifier, instead of `labels`. */
  classifier?: string;
  instructions?: string;
  multiLabel?: boolean;
  threshold?: number;
  model?: string;
}

export interface Job {
  id: string;
  kind: string;
  status: JobStatus;
  done: number;
  total: number | null;
  failed: number;
  percent: number | null;
  etaS: number | null;
  error: string | null;
  /** The kind's final summary (classify: { items, ok, errors, by_label }). */
  result: any;
  metadata: Record<string, unknown> | null;
  createdAt: string | null;
  startedAt: string | null;
  finishedAt: string | null;
  /** { url, events, has_secret, deliveries } - the secret is never returned. */
  webhook: Record<string, any> | null;
  /** succeeded, failed or cancelled. */
  finished: boolean;
  ok: boolean;
  raw: Record<string, any>;
}

export interface JobItem {
  index: number;
  error: string | null;
  ok: boolean;
  /** Typed for classify / score jobs; null for failed rows and other kinds (use `raw`). */
  result: Classification | ScoreResult | null;
  raw: Record<string, any>;
}

export interface WaitJobOptions {
  timeoutMs?: number;
  pollMs?: number;
  onProgress?: (job: Job) => void;
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
  submitJob(kind: string, payload: Record<string, any>, opts?: JobOptions): Promise<Job>;
  classifyJob(inputs: unknown[], labels?: Labels | null, opts?: ClassifyJobOptions): Promise<Job>;
  job(id: string): Promise<Job>;
  jobs(opts?: { status?: JobStatus; kind?: string; limit?: number; offset?: number }): Promise<Job[]>;
  cancelJob(id: string): Promise<Job>;
  deleteJob(id: string): Promise<{ deleted: string }>;
  waitJob(id: string, opts?: WaitJobOptions): Promise<Job>;
  jobResults(id: string, opts?: { pageSize?: number; offset?: number }): AsyncGenerator<JobItem, void, undefined>;
  classifier(name: string): Classifier;
  listClassifiers(): Promise<Array<Record<string, any>>>;
  saveClassifier(name: string, def?: ClassifierDefinition): Promise<Record<string, any>>;
  getClassifier(name: string): Promise<Record<string, any>>;
  deleteClassifier(name: string): Promise<{ deleted: string }>;
}

export function parseClassification(body: Record<string, any>, requestId?: string | null): Classification;
export function parseScore(body: Record<string, any>, requestId?: string | null): ScoreResult;
export function parseJob(body: Record<string, any>): Job;
export function parseJobItem(row: Record<string, any>, kind?: string): JobItem;
export function bytesToDataUrl(bytes: Uint8Array | ArrayBuffer, mime?: string): string;
