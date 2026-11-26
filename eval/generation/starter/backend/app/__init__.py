"""Taskboard backend.

The package is organised by responsibility:

* :mod:`app.config`     - settings read from the environment;
* :mod:`app.db`         - SQLite connections, the ``get_db`` dependency and migrations;
* :mod:`app.security`   - password hashing, access tokens, ``get_current_user``;
* :mod:`app.errors`     - :class:`~app.errors.ApiError` and the JSON error format;
* :mod:`app.pagination` - ``limit``/``offset`` parameters and paged responses;
* :mod:`app.routers`    - one router per feature;
* :mod:`app.main`       - the application factory that wires everything together.
"""

__version__ = "0.3.0"
