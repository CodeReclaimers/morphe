# SolidWorks COM Late-Binding Quirks

This document captures quirks and workarounds discovered when using SolidWorks API via Python's `win32com.client` with late binding (dynamic dispatch).

## Core Issues

### 1. Method vs Property Ambiguity
Late-bound COM exposes some methods as properties. Always check if an attribute is callable:

```python
value = obj.SomeAttribute
if callable(value):
    value = value()
```

### 2. Different COM Wrapper Objects
COM returns **different Python wrapper objects** for the same underlying COM object. You cannot use `id()` or `is` to compare objects.

**Problem:**
```python
segments = sketch.GetSketchSegments
entity_from_relation = relation.GetEntities[0]
# id(segments[0]) != id(entity_from_relation)  # Even if same object!
```

**Solution:** Match by geometric properties (length, radius, coordinates):
```python
# Build lookup by length
length_to_id = {}
for seg in segments:
    length_key = round(seg.GetLength, 10)
    length_to_id[length_key] = prim_id

# Match entity by length
entity_length = round(entity.GetLength, 10)
matched_id = length_to_id.get(entity_length)
```

### 3. Feature Iteration
`FirstFeature` and `GetNextFeature` don't work with late-bound COM.

**Problem:**
```python
feat = doc.FirstFeature()  # Error: "Member not found"
```

**Solution:** Use `FeatureByPositionReverse` with index:
```python
fm = doc.FeatureManager
feature_count = fm.GetFeatureCount(True)
for i in range(feature_count):
    feat = doc.FeatureByPositionReverse(i)
    if feat is not None:
        # Process feature
```

### 4. Sketch Access via GetSpecificFeature2
Accessing sketch methods requires the `GetSpecificFeature2` wrapper.

**Problem:**
```python
feat = find_sketch_feature("Sketch1")
segments = feat.GetSketchSegments()  # Error
```

**Solution:** Access `GetSpecificFeature2` WITHOUT calling it:
```python
sketch_obj = feat.GetSpecificFeature2  # Note: no parentheses!
segments = sketch_obj.GetSketchSegments
if callable(segments):
    segments = segments()
```

### 5. Constraint/Relation Access
`GetSketchRelations()` on the sketch object doesn't work.

**Problem:**
```python
relations = sketch.GetSketchRelations()  # Error or returns None
```

**Solution:** Iterate segments and get relations from each:
```python
segments = sketch_obj.GetSketchSegments
for segment in segments:
    relations = segment.GetRelations
    if callable(relations):
        relations = relations()
    for rel in relations:
        rel_type = rel.GetRelationType
        if callable(rel_type):
            rel_type = rel_type()
```

### 6. Dimension Access
`GetDisplayDimensions()` doesn't work via late-bound COM. Dimensional constraints (diameter, length, radius, angle) cannot currently be read.

**Workaround:** For import, we recreate geometry at the correct dimensions rather than applying dimension constraints (which would open blocking dialogs anyway).

## Constraint Type Constants

From `swConstraintType_e` (SolidWorks 2024 API):
https://help.solidworks.com/2024/english/api/swconst/swConstraintType_e.html

| Value | Name | Canonical Mapping |
|-------|------|-------------------|
| 4 | HORIZONTAL | HORIZONTAL |
| 5 | VERTICAL | VERTICAL |
| 6 | TANGENT | TANGENT |
| 7 | PARALLEL | PARALLEL |
| 8 | PERPENDICULAR | PERPENDICULAR |
| 9 | COINCIDENT | COINCIDENT |
| 10 | CONCENTRIC | CONCENTRIC |
| 11 | SYMMETRIC | SYMMETRIC |
| 12 | ATMIDDLE | MIDPOINT |
| 14 | SAMELENGTH | EQUAL |
| 17 | FIXED | FIXED |
| 27 | COLINEAR | COLLINEAR |
| 28 | CORADIAL | EQUAL (radius) |

**Dimensional constraints** (1=DISTANCE, 2=ANGLE, 3=RADIUS, 15=DIAMETER) are not accessible via `GetRelations` - they require `GetDisplayDimensions` which doesn't work with late-bound COM.

## Best Practices

1. **Always use `_get_com_result` helper** for property/method access
2. **Build property-based lookups** for entity matching (length, radius)
3. **Use `FeatureByPositionReverse`** for feature iteration
4. **Access sketch via `feat.GetSpecificFeature2`** (uncalled)
5. **Iterate segments for relations** instead of calling sketch-level methods
6. **Handle exceptions gracefully** - many operations fail silently in late-bound COM
