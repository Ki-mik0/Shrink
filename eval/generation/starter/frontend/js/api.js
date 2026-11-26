// Talking to the Taskboard API.
//
// Every request from the frontend goes through apiFetch(). It
//   * prefixes nothing: pass absolute API paths such as "/api/health";
//   * sends and parses JSON;
//   * adds "Authorization: Bearer <token>" when a token is stored;
//   * turns error responses into an ApiError that carries the server's error
//     code (see backend/app/errors.py for the format), so callers can react to
//     specific failures, e.g. `if (error.code === "invalid_token") ...`;
//   * forgets a token the server rejected, so the user is asked to sign in
//     again instead of seeing the same error on every request.

/** Key under which the access token is kept in localStorage. */
export const TOKEN_KEY = "taskboard.token";

/** An error answered by the API, or a network failure (code "network_error"). */
export class ApiError extends Error {
  /**
   * @param {number} status HTTP status, 0 for network failures.
   * @param {string} code Stable error code from the server.
   * @param {string} message Human-readable explanation.
   * @param {Array<{field: string, problem: string}>} [details] Validation problems.
   */
  constructor(status, code, message, details = []) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

/** Store the token returned by the sign-in endpoint. */
export function saveToken(token) {
  localStorage.setItem(TOKEN_KEY, token);
}

/** The stored token, or null when the user is not signed in. */
export function loadToken() {
  return localStorage.getItem(TOKEN_KEY);
}

/** Forget the token (sign out). */
export function clearToken() {
  localStorage.removeItem(TOKEN_KEY);
}

/** True when a token is stored. The server may still reject it if it expired. */
export function isSignedIn() {
  return loadToken() !== null;
}

/**
 * Call the API and return the parsed JSON body (null for 204 No Content).
 *
 * @param {string} path Absolute API path, e.g. "/api/health".
 * @param {object} [options]
 * @param {string} [options.method="GET"] HTTP method.
 * @param {object} [options.body] Value sent as JSON.
 * @param {object} [options.query] Query parameters; undefined/null values are skipped.
 * @param {boolean} [options.auth=true] Attach the stored token if there is one.
 * @returns {Promise<any>}
 * @throws {ApiError}
 */
export async function apiFetch(path, { method = "GET", body, query, auth = true } = {}) {
  const url = new URL(path, window.location.origin);
  for (const [key, value] of Object.entries(query ?? {})) {
    if (value !== undefined && value !== null) url.searchParams.set(key, String(value));
  }

  const headers = { Accept: "application/json" };
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const token = auth ? loadToken() : null;
  if (token) headers.Authorization = `Bearer ${token}`;

  let response;
  try {
    response = await fetch(url, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new ApiError(0, "network_error", "The server cannot be reached. Check your connection.");
  }

  if (response.status === 204) return null;

  let data = null;
  const text = await response.text();
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      throw new ApiError(response.status, "bad_response", "The server sent an unreadable answer.");
    }
  }

  if (!response.ok) {
    const error = data?.error ?? {};
    if (response.status === 401) clearToken();
    throw new ApiError(
      response.status,
      error.code ?? "http_error",
      error.message ?? `Request failed with status ${response.status}.`,
      error.details ?? [],
    );
  }
  return data;
}
