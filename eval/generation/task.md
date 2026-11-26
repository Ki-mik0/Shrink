Add user accounts and personal todo lists to Taskboard, from the backend to the
frontend. Follow the conventions in README.md and reuse the helpers that already
exist (get_db, MIGRATIONS, hash_password, verify_password, create_access_token,
get_current_user, ApiError and its helpers, page_params, paginate, utc_now,
apiFetch, saveToken, loadToken, clearToken, el, clear, showMessage).

Backend (FastAPI, SQLite):

1. A migration that creates the table `todos` (id, user_id referencing users,
   title, done, created_at, updated_at).
2. Router `app/routers/auth.py` with prefix `/api/auth`:
   - `POST /register` with `{"username", "password"}`: 201 and `{"id", "username",
     "created_at"}`; username 3-32 characters of letters, digits, `_` or `-`;
     weak password -> 400 `weak_password`; taken username -> 409 `username_taken`.
   - `POST /login` with `{"username", "password"}`: 200 and `{"access_token",
     "token_type": "bearer", "expires_in"}`; wrong credentials -> 401
     `invalid_credentials` (same answer for unknown user and wrong password).
     Record `login` and `login_failed` in the audit log.
   - `GET /me`: the signed-in user.
3. Router `app/routers/todos.py` with prefix `/api/todos`, every endpoint for
   the signed-in user only:
   - `GET ""`: the user's todos, newest first, paged with page_params and
     paginate; optional filter `?done=true|false`.
   - `POST ""` with `{"title"}` (1-200 characters after trimming): 201 and the todo.
   - `GET /{id}`, `PATCH /{id}` with `{"title"?, "done"?}`, `DELETE /{id}` (204).
   - Another user's todo or a missing one -> 404 `todo_not_found`.
   A todo in JSON is `{"id", "title", "done", "created_at", "updated_at"}`.
4. Register both routers in `create_app()`.

Frontend (plain HTML/CSS/JS, no frameworks):

5. `frontend/login.html` + `frontend/js/login.js`: sign-in and registration
   forms; after signing in, store the token and go to the todo page.
6. `frontend/todos.html` + `frontend/js/todos.js`: redirect to the sign-in page
   when not signed in; show the user's name and a sign-out button; list todos
   with paging (Previous/Next) and a filter (all/open/done); add a todo; toggle
   done; rename; delete with confirmation. Show API errors with showMessage.
7. Update the navigation in `frontend/index.html` to link to these pages.

Return every new or changed file in full, each one as:

## File: <path relative to the repository root>
```<language>
<complete file content>
```

Do not explain the code outside these blocks.
