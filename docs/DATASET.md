# Dataset

Raw data is **not** included in the repository (size, and it reveals real access points and locations).
Place it in `data/raw/`, which is git-ignored.

## Expected Folder Layout

```
data/raw/
├── device_1/
│   ├── room1_1.csv
│   ├── room1_2.csv
│   └── ...
├── device_2/
├── device_3/
└── device_4/
```

File naming convention: `<location>_<reference_point>.csv`, for example `room3_7.csv`
(location `room3`, reference point 7). Locations are upper-cased by the notebook.

## CSV Schema

| Device | SSID column | MAC column | Signal column |
|---|---|---|---|
| `device_1` | `SSID` | `Target MAC Address` | `Sinyal Terbaru (dBm)` |
| `device_2` - `device_4` | `Name` | `Address` | `Signal Level` |

> `Sinyal Terbaru (dBm)` is the raw column name written by the `device_1` logging tool, so it is kept
> exactly as exported. If your tool writes different headers, edit the column mapping in notebook section 2.

Only rows whose SSID contains the configured `ssid_keyword` are used.

## Processed Dataset

Columns: 12 RSSI columns (one per BSSID, lowercase) + `reference_point`, `location`, `device`.
APs that were not detected are filled with `-100`.

Summary of the dataset the reported results were produced from: 360 fingerprints =
10 locations x 9 reference points x 4 devices.

## Tips

- If you publish sample data, use synthetic values or a handful of rows with BSSIDs replaced by the
  placeholder addresses from `config/access_points.example.json`.
