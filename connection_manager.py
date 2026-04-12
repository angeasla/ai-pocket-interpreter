import logging
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages active WebSocket connections and broadcasting."""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("Client connected. Total clients: %d", len(self.active_connections))

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a disconnected client from the active list."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info("Client disconnected. Total clients: %d", len(self.active_connections))

    async def broadcast(self, payload: dict) -> None:
        """Send JSON payload to all connected clients with per-client error handling.

        Handles disconnected clients gracefully by catching exceptions,
        removing dead connections, and continuing to broadcast to remaining clients.
        Silently handles zero-client broadcasts.
        """
        dead_connections: list[WebSocket] = []

        for connection in self.active_connections[:]:  # iterate over a copy
            try:
                await connection.send_json(payload)
            except Exception:
                logger.warning("Failed to send to client, removing connection.")
                dead_connections.append(connection)

        for connection in dead_connections:
            self.disconnect(connection)
