// The error body of every non-2xx answer (docs/API.md, Errors).

// ---------- Errors ----------

export interface ApiErrorBody {
  /** `hint` is always present on the server, `null` when there is nothing to add. */
  error: { code: string; message: string; hint: string | null };
}
