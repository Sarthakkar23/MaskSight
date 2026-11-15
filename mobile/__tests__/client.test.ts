import { ApiError, createClient } from '../src/api/client';

const summary = {
  id: 7,
  created_at: '2026-11-15T10:00:00+00:00',
  category: 'bottle',
  anomaly_score: 25.6,
  threshold: 13.6,
  is_defective: true,
  defect_area_fraction: 0.17,
  latency_ms: 61,
  filename: null,
};

function respond(status: number, body: unknown): jest.Mock {
  return jest.fn().mockResolvedValue({ ok: status < 400, status, json: async () => body });
}

describe('API client', () => {
  it('parses typed lists and builds URLs from the base', async () => {
    const fetchImpl = respond(200, [summary]);
    const api = createClient({ baseUrl: 'http://host:8000/', fetchImpl });
    await expect(api.inspections(20)).resolves.toEqual([summary]);
    expect(fetchImpl.mock.calls[0][0]).toBe('http://host:8000/inspections?limit=20');
  });

  it('uploads the picked file as multipart form data', async () => {
    const detail = { ...summary, image_png_base64: 'a', mask_png_base64: 'b', mask_layer_png_base64: 'c' };
    const fetchImpl = respond(200, detail);
    const api = createClient({ baseUrl: 'http://host:8000', fetchImpl });
    const res = await api.predict('metal nut', { uri: 'file:///x.jpg', name: 'x.jpg', mimeType: 'image/jpeg' });
    expect(res.mask_layer_png_base64).toBe('c');
    const [url, init] = fetchImpl.mock.calls[0];
    expect(url).toBe('http://host:8000/predict?category=metal%20nut');
    expect(init.method).toBe('POST');
    expect(init.body).toBeInstanceOf(FormData);
  });

  it('surfaces the server error message', async () => {
    const api = createClient({ baseUrl: 'http://h', fetchImpl: respond(404, { detail: "no model for 'x'" }) });
    await expect(api.categories()).rejects.toEqual(new ApiError("no model for 'x'", 404));
  });

  it('reports an unreachable server and rejects malformed responses', async () => {
    const down = createClient({ baseUrl: 'http://h', fetchImpl: jest.fn().mockRejectedValue(new TypeError('fail')) });
    await expect(down.categories()).rejects.toThrow('Server unreachable at http://h');

    const bad = createClient({ baseUrl: 'http://h', fetchImpl: respond(200, [{ ...summary, id: 'seven' }]) });
    await expect(bad.inspections()).rejects.toThrow('Unexpected inspection from the server (id)');
  });
});
