"""Qualitative fixpoint algorithms for MDPs.

Implements the operators and provides a generic
:class:`FixpointIterator` and :func:`visualise_iterations`

Three operators are defined, each mirroring one qualitative reachability property:

* :func:`psi_spos`    — LFP for :math:`S_{\\text{pos}}`   (possible max reach)
* :func:`psi_sposmin` — LFP for :math:`S_{\\text{pos}}^{\\min}` (possible min reach)
* :func:`psi_smaxas`  — GFP for :math:`S_{\\text{as}}^{\\max}` (almost-sure max reach)
"""

from collections.abc import Callable

import stormvogel.model as model

_Operator = Callable[
    [frozenset[model.State], model.Model, frozenset[model.State]],
    frozenset[model.State],
]


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


def psi_spos(
    X: frozenset[model.State],
    mdp: model.Model,
    target: frozenset[model.State],
) -> frozenset[model.State]:
    r"""Operator for :math:`S_{\text{pos}}` (possible max reachability).

    A state is added when it has *some* action with *some* successor in *X*.
    LFP starting from *target* equals :math:`S_{\text{pos}}`.

    .. math::

       \Psi_{\max>0}(X) =
         T \cup \{s \mid \exists a \in \mathrm{En}(s).\;
                        \exists s' \in X.\; \delta(s,a)(s') > 0\}
    """
    new: set[model.State] = set(target)
    for s in mdp.states:
        if s in new:
            continue
        for _action, branch in s.choices:
            if any(t in X for _, t in branch):
                new.add(s)
                break
    return frozenset(new)


def psi_sposmin(
    X: frozenset[model.State],
    mdp: model.Model,
    target: frozenset[model.State],
) -> frozenset[model.State]:
    r"""Operator for :math:`S_{\text{pos}}^{\min}` (possible min reachability).

    A state is added when *every* action has *some* successor in *X*.
    LFP starting from *target* equals :math:`S_{\text{pos}}^{\min}`.

    .. math::

       \Psi_{\min>0}(X) =
         T \cup \{s \mid \forall a \in \mathrm{En}(s).\;
                        \exists s' \in X.\; \delta(s,a)(s') > 0\}
    """
    new: set[model.State] = set(target)
    for s in mdp.states:
        if s in new:
            continue
        if all(any(t in X for _, t in branch) for _action, branch in s.choices):
            new.add(s)
    return frozenset(new)


def psi_sminas(
    X: frozenset[model.State],
    mdp: model.Model,
    target: frozenset[model.State],
) -> frozenset[model.State]:
    r"""Operator for :math:`S_{\text{as}}^{\min}` (almost-sure min reachability).

    A non-target state is *kept* when *every* action keeps all its successors
    inside *X*.  If any action can leave *X* with positive probability the
    adversarial scheduler will pick it, so the state is removed.

    .. math::

       \Psi_{\min=1}(X) =
         T \cup \{s \in X \mid \forall a \in \mathrm{En}(s).\;
                               \forall s' \in \mathrm{supp}(\delta(s,a)).\; s' \in X\}

    .. note::

       Must be initialised from :math:`S_{\text{pos}}^{\min}` (see
       :func:`sminas`), not from :math:`S`, because :math:`S` is a trivial
       fixed point of this operator.
    """
    new: set[model.State] = set(target)
    for s in mdp.states:
        if s in target:
            continue
        if s in X and all(
            all(t in X for _, t in branch) for _action, branch in s.choices
        ):
            new.add(s)
    return frozenset(new)


def psi_smaxas(
    X: frozenset[model.State],
    mdp: model.Model,
    target: frozenset[model.State],
) -> frozenset[model.State]:
    r"""Operator for :math:`S_{\text{as}}^{\max}` (almost-sure max reachability).

    A non-target state is *kept* when it has *some* action all of whose
    successors are in *X*.  GFP starting from all states equals
    :math:`S_{\text{as}}^{\max}`.

    .. math::

       \Psi_{\max=1}(X) =
         T \cup \{s \mid \exists a \in \mathrm{En}(s).\;
                        \forall s' \in \mathrm{supp}(\delta(s,a)).\; s' \in X\}
    """
    new: set[model.State] = set(target)
    for s in mdp.states:
        if s in target:
            continue
        for _action, branch in s.choices:
            if all(t in X for _, t in branch):
                new.add(s)
                break
    return frozenset(new)


# ---------------------------------------------------------------------------
# Generic fixpoint iterator
# ---------------------------------------------------------------------------


class FixpointIterator:
    """Iterate a set-valued operator to its fixpoint.

    :param operator: One of :func:`psi_spos`, :func:`psi_sposmin`,
        :func:`psi_smaxas`, or any compatible callable.
    :param initial: Starting set (use *target* for LFP, all states for GFP).
    :param mdp: The MDP.
    :param target: The fixed target set passed to the operator each step.
    """

    def __init__(
        self,
        operator: _Operator,
        initial: frozenset[model.State],
        mdp: model.Model,
        target: frozenset[model.State],
    ) -> None:
        self._op = operator
        self._current = initial
        self._mdp = mdp
        self._target = target
        self._converged = False

    @property
    def current(self) -> frozenset[model.State]:
        """The current set."""
        return self._current

    def has_converged(self) -> bool:
        """True after a step where the set did not change."""
        return self._converged

    def step(self) -> frozenset[model.State]:
        """Apply the operator once and return the new set."""
        new = self._op(self._current, self._mdp, self._target)
        self._converged = new == self._current
        self._current = new
        return self._current


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def spos(
    mdp: model.Model,
    target_states: list[model.State] | set[model.State],
) -> FixpointIterator:
    """Return a :class:`FixpointIterator` for :math:`S_{\\text{pos}}` (LFP)."""
    t = frozenset(target_states)
    return FixpointIterator(psi_spos, t, mdp, t)


def sposmin(
    mdp: model.Model,
    target_states: list[model.State] | set[model.State],
) -> FixpointIterator:
    """Return a :class:`FixpointIterator` for :math:`S_{\\text{pos}}^{\\min}` (LFP)."""
    t = frozenset(target_states)
    return FixpointIterator(psi_sposmin, t, mdp, t)


def smaxas(
    mdp: model.Model,
    target_states: list[model.State] | set[model.State],
) -> FixpointIterator:
    """Return a :class:`FixpointIterator` for :math:`S_{\\text{as}}^{\\max}` (GFP)."""
    t = frozenset(target_states)
    return FixpointIterator(psi_smaxas, frozenset(mdp.states), mdp, t)


def sminas(
    mdp: model.Model,
    target_states: list[model.State] | set[model.State],
) -> FixpointIterator:
    r"""Return a :class:`FixpointIterator` for :math:`S_{\text{as}}^{\min}` (GFP).

    Initialises from :math:`S_{\text{pos}}^{\min}` — the result of running
    :func:`sposmin` to convergence — so that states the minimiser can trap
    forever are excluded before the GFP begins.  Starting from the full state
    space :math:`S` would give a trivial fixed point.
    """
    t = frozenset(target_states)
    sposmin_it = FixpointIterator(psi_sposmin, t, mdp, t)
    while not sposmin_it.has_converged():
        sposmin_it.step()
    return FixpointIterator(psi_sminas, sposmin_it.current, mdp, t)


# ---------------------------------------------------------------------------
# Convenience: run to fixpoint and return the final set directly
# ---------------------------------------------------------------------------


def _run(it: FixpointIterator) -> frozenset[model.State]:
    while not it.has_converged():
        it.step()
    return it.current


def compute_spos(
    mdp: model.Model,
    target_states: list[model.State] | set[model.State],
) -> frozenset[model.State]:
    r"""Return :math:`S_{\text{pos}}` directly (runs LFP to convergence)."""
    return _run(spos(mdp, target_states))


def compute_sposmin(
    mdp: model.Model,
    target_states: list[model.State] | set[model.State],
) -> frozenset[model.State]:
    r"""Return :math:`S_{\text{pos}}^{\min}` directly (runs LFP to convergence)."""
    return _run(sposmin(mdp, target_states))


def compute_smaxas(
    mdp: model.Model,
    target_states: list[model.State] | set[model.State],
) -> frozenset[model.State]:
    r"""Return :math:`S_{\text{as}}^{\max}` directly (runs GFP to convergence)."""
    return _run(smaxas(mdp, target_states))


def compute_sminas(
    mdp: model.Model,
    target_states: list[model.State] | set[model.State],
) -> frozenset[model.State]:
    r"""Return :math:`S_{\text{as}}^{\min}` directly (runs GFP to convergence)."""
    return _run(sminas(mdp, target_states))


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def visualise_iterations(
    snapshots: list[frozenset[model.State]],
    mdp: model.Model,
    highlight: bool = True,
):
    """Return a pandas DataFrame showing set membership at each iteration.

    Rows are states; column *i* shows whether that state was in the set after
    step *i*.  Pass the list of sets returned by successive :meth:`FixpointIterator.step`
    calls (including the initial set if desired).

    :param snapshots: Ordered list of state sets, one per iteration.
    :param mdp: The MDP (used to fix row order via ``sorted_states``).
    :param highlight: If True, colour ``True`` cells green and ``False``
        cells red via a pandas :class:`~pandas.io.formats.style.Styler`.
    :returns: A styled or plain :class:`pandas.DataFrame`.
    """
    import pandas as pd

    states = list(mdp.sorted_states)
    data = {
        f"iter {i}": [s in snap for s in states] for i, snap in enumerate(snapshots)
    }
    df = pd.DataFrame(data, index=[s.friendly_name or str(s.state_id) for s in states])
    df.index.name = "state"

    if highlight:
        return df.style.map(
            lambda v: "background-color: #c6efce" if v else "background-color: #ffc7ce"
        )
    return df
