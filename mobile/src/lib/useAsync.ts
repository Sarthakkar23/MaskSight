import { useCallback, useEffect, useRef, useState } from 'react';

export type AsyncState<T> =
  | { status: 'loading'; data?: T }
  | { status: 'error'; error: string; data?: T }
  | { status: 'success'; data: T };

const message = (err: unknown) => (err instanceof Error ? err.message : 'Something went wrong');

/** Runs `load` on mount (and on `reload`), tracking loading / error / success. Stale results are ignored. */
export function useAsync<T>(load: () => Promise<T>, deps: readonly unknown[] = []) {
  const [state, setState] = useState<AsyncState<T>>({ status: 'loading' });
  const run = useRef(0);

  const reload = useCallback(() => {
    const id = ++run.current;
    setState((s) => ({ status: 'loading', data: s.data }));
    load().then(
      (data) => id === run.current && setState({ status: 'success', data }),
      (err: unknown) => id === run.current && setState((s) => ({ status: 'error', error: message(err), data: s.data })),
    );
  }, deps);

  useEffect(() => {
    reload();
  }, [reload]);

  return { state, reload };
}
