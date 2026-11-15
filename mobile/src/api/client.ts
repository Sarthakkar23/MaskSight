import type { Category, InspectionDetail, InspectionSummary, PickedImage } from './types';

/** Error with a message that can be shown to the user as-is. */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number | null = null,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

type Fetch = typeof fetch;

export interface ClientOptions {
  baseUrl: string;
  fetchImpl?: Fetch;
  timeoutMs?: number;
}

const isObject = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null;

function check<T>(value: unknown, keys: Record<string, string>, what: string): T {
  if (!isObject(value)) throw new ApiError(`Unexpected ${what} from the server`);
  for (const [key, type] of Object.entries(keys)) {
    const actual = value[key] === null ? 'null' : typeof value[key];
    if (!type.split('|').includes(actual)) throw new ApiError(`Unexpected ${what} from the server (${key})`);
  }
  return value as T;
}

const SUMMARY_KEYS = {
  id: 'number',
  created_at: 'string',
  category: 'string',
  anomaly_score: 'number',
  threshold: 'number',
  is_defective: 'boolean',
  defect_area_fraction: 'number',
  latency_ms: 'number',
  filename: 'string|null',
};
const DETAIL_KEYS = {
  ...SUMMARY_KEYS,
  image_png_base64: 'string',
  mask_png_base64: 'string',
  mask_layer_png_base64: 'string',
};

export const parseCategory = (v: unknown) =>
  check<Category>(v, { name: 'string', model: 'string', image_threshold: 'number' }, 'category');
export const parseSummary = (v: unknown) => check<InspectionSummary>(v, SUMMARY_KEYS, 'inspection');
export const parseDetail = (v: unknown) => check<InspectionDetail>(v, DETAIL_KEYS, 'inspection');

function parseList<T>(v: unknown, parse: (item: unknown) => T): T[] {
  if (!Array.isArray(v)) throw new ApiError('Unexpected list from the server');
  return v.map(parse);
}

export function createClient({ baseUrl, fetchImpl = fetch, timeoutMs = 30000 }: ClientOptions) {
  const root = baseUrl.replace(/\/+$/, '');

  async function request(path: string, init?: RequestInit): Promise<unknown> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    let res: Response;
    try {
      res = await fetchImpl(`${root}${path}`, { ...init, signal: controller.signal });
    } catch (err) {
      const aborted = err instanceof Error && err.name === 'AbortError';
      throw new ApiError(aborted ? 'The server took too long to answer' : `Server unreachable at ${root}`);
    } finally {
      clearTimeout(timer);
    }
    let body: unknown = null;
    try {
      body = await res.json();
    } catch {
      // non-JSON body (e.g. a proxy error page); handled below
    }
    if (!res.ok) {
      const detail = isObject(body) && typeof body.detail === 'string' ? body.detail : `HTTP ${res.status}`;
      throw new ApiError(detail, res.status);
    }
    return body;
  }

  return {
    baseUrl: root,
    categories: async () => parseList(await request('/categories'), parseCategory),
    inspections: async (limit = 50) => parseList(await request(`/inspections?limit=${limit}`), parseSummary),
    inspection: async (id: number) => parseDetail(await request(`/inspections/${id}`)),
    predict: async (category: string, image: PickedImage) => {
      const form = new FormData();
      if (image.file) {
        form.append('image', image.file, image.name);
      } else {
        // React Native's FormData accepts { uri, name, type } and streams the file from disk.
        form.append('image', { uri: image.uri, name: image.name, type: image.mimeType } as unknown as Blob);
      }
      const body = await request(`/predict?category=${encodeURIComponent(category)}`, {
        method: 'POST',
        body: form,
      });
      return parseDetail(body);
    },
  };
}

export type ApiClient = ReturnType<typeof createClient>;
