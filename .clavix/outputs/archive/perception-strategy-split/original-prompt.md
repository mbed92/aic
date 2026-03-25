# Original Prompt (Extracted from Conversation)

*Extracted by Clavix on 2026-03-25. See optimized-prompt.md for enhanced version.*

Rozbij `PerceptionStrategy` na dwa osobne interfejsy: jeden do generowania pointclouda z obrazów RGB (`PointCloudStrategy`), drugi do lokalizacji portu w pointcloudzie (`BoardLocalizationStrategy`). Motywacja to wymienność — system może używać SGBM albo modeli AI do generowania pointclouda, niezależnie od metody lokalizacji.

Format pośredni między strategiami to `open3d.geometry.PointCloud`. Oba interfejsy są wymienialne. Orkiestrację zapewnia nowy `CompositePerceptionStrategy` który implementuje istniejący `PerceptionStrategy` — dzięki temu `CompositeInsertCableStrategy` nie zmienia się. `SGBMICPPerception` (stub TODO) zostaje podzielony na dwa osobne stuby: `SGBMPointCloud` i `ICPBoardLocalization`.

---
*Extracted by Clavix on 2026-03-25. See optimized-prompt.md for enhanced version.*
