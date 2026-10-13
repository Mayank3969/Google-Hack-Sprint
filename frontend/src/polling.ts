import type { Adapter, Message } from './api';
// A serial loop owns the one outstanding request. Server snapshots replace the
// list, preserving intentional identical messages without append duplication.
export function pollMessages(api: Adapter, room: string, onMessages: (messages: Message[]) => void, onError: (message: string) => void, documentRef: Pick<Document, 'hidden' | 'addEventListener' | 'removeEventListener'> = document) {
  let stopped = false, pending = false;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const controller = new AbortController();
  async function refresh() {
    if (stopped || pending || documentRef.hidden) return;
    clearTimeout(timer); pending = true;
    try { const result = await api.messages(room, controller.signal); if (!stopped) { onMessages(result); onError(''); } }
    catch (error) { if (!stopped) onError(error instanceof Error ? error.message : 'Conversation sync failed.'); }
    finally { pending = false; if (!stopped && !documentRef.hidden) timer = setTimeout(refresh, 2000); }
  }
  const visibility = () => { clearTimeout(timer); if (!documentRef.hidden) void refresh(); };
  documentRef.addEventListener('visibilitychange', visibility);
  void refresh();
  return { refresh, stop() { stopped = true; clearTimeout(timer); controller.abort(); documentRef.removeEventListener('visibilitychange', visibility); } };
}
