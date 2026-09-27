"use client";

import { useCallback, useEffect, useRef } from "react";
import { useRunStore } from "@/store/useRunStore";
import type { ServerEvent, Variant } from "@/lib/types";

const WS_URL = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000/ws/generate";
const RECONNECT_DELAY_MS = 1500;

export function useGenerationSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const setConnection = useRunStore((s) => s.setConnection);
  const applyServerEvent = useRunStore((s) => s.applyServerEvent);

  useEffect(() => {
    let cancelled = false;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    function connect() {
      if (cancelled) return;
      setConnection("connecting");
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => setConnection("connected");
      ws.onclose = () => {
        setConnection("disconnected");
        if (!cancelled) reconnectTimer = setTimeout(connect, RECONNECT_DELAY_MS);
      };
      ws.onerror = () => ws.close();
      ws.onmessage = (msg) => {
        try {
          const event = JSON.parse(msg.data) as ServerEvent;
          applyServerEvent(event);
        } catch {
          // ignore malformed frames
        }
      };
    }

    connect();
    return () => {
      cancelled = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      wsRef.current?.close();
    };
  }, [setConnection, applyServerEvent]);

  const startGist = useCallback((budget: number, variant: Variant | null, contextLen: number) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    useRunStore.getState().startGistRun(budget, variant, contextLen);
    ws.send(JSON.stringify({ type: "start_gist_run", budget, variant, context_len: contextLen }));
  }, []);

  const startBatch = useCallback((contextLen: number) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    useRunStore.getState().startBatchRun(contextLen);
    ws.send(JSON.stringify({ type: "start_batch_run", context_len: contextLen }));
  }, []);

  const cancel = useCallback(() => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ type: "cancel_run" }));
  }, []);

  return { startGist, startBatch, cancel };
}
