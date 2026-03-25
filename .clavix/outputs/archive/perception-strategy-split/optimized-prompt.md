# Optimized Prompt (Clavix Enhanced)

*Optimized by Clavix on 2026-03-25. This version is ready for implementation.*

Refactor `strategies/perception/` w `romanelovect_policy`: rozbij `PerceptionStrategy` na trzy komponenty zachowując pełną backward compatibility z `CompositeInsertCableStrategy`.

**Nowe ABC** (`strategies/perception/base.py`):
- `PointCloudStrategy` — `generate(task: Task, get_observation: GetObservationCallback) -> open3d.geometry.PointCloud | None`
- `BoardLocalizationStrategy` — `estimate_port_pose(task: Task, point_cloud: open3d.geometry.PointCloud) -> Pose | None`

**Nowa klasa kompozytowa** (`strategies/perception/composite.py`):
`CompositePerceptionStrategy(PerceptionStrategy)` — przyjmuje `PointCloudStrategy` i `BoardLocalizationStrategy` przez konstruktor. `estimate_port_pose()` wywołuje je sekwencyjnie: najpierw `generate()`, jeśli wynik nie jest `None` przekazuje do `estimate_port_pose()` lokalizatora.

**Podział stuba** (`strategies/perception/sgbm_icp.py` → dwa pliki):
- `SGBMPointCloud(PointCloudStrategy)` — implementuje `generate()` z `raise NotImplementedError`
- `ICPBoardLocalization(BoardLocalizationStrategy)` — implementuje `estimate_port_pose()` z `raise NotImplementedError`

**Niezmienione:** `PerceptionStrategy` ABC i `CompositeInsertCableStrategy` — zero modyfikacji.

Invariant: podmiana `SGBMPointCloud` na `AIDepthPointCloud` wymaga zmiany wyłącznie w konstruktorze `RomanelovectPolicy`.

---

## Optimization Improvements Applied

1. **[Structured]** — Podzielono na: nowe ABC → klasa kompozytowa → podział stuba → niezmienione części. Każda sekcja to jeden plik.
2. **[Clarified]** — Jawnie wymieniono które pliki się zmieniają, a które nie (`CompositeInsertCableStrategy` = zero zmian).
3. **[Completeness]** — Dodano sygnaturę `generate()` z pełnymi typami, której brakowało w rozmowie.
4. **[Actionability]** — Dodano invariant podmiany jako kryterium weryfikacji poprawności implementacji.
5. **[Efficiency]** — Usunięto dyskusję o zasadach SOLID; zostawiono tylko decyzje.

---
*Optimized by Clavix on 2026-03-25. This version is ready for implementation.*
