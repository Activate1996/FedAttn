Portable plotting data for the Fig. 2 and Fig. 3 representation-deviation experiments.
representation_deviation_summary.json contains panel arrays, per-point statistics, and sample counts.
representation_deviation_points.csv contains one row per configuration;
representation_deviation_per_sample.csv contains the individual measurements.
get_representation_curve(..., figure=2 or 3, model=..., split_way=..., x_values=x)
in export_representation_deviation.py reads means in the requested x order.
Missing or incomplete points return NaN. sample_plan.json records the shared dataset row indices.
