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

**Solution:** Use index-based mapping built during segment iteration:
```python
# Build index-to-ID mapping during primitive export
segment_index_to_id = []
for seg_idx, segment in enumerate(segments):
    prim = export_segment(segment)
    segment_index_to_id[seg_idx] = prim.id

# Use index-based lookup during constraint export
for seg_idx, segment in enumerate(segments):
    segment_id = segment_index_to_id[seg_idx]  # Reliable!
```

**Why length-based matching fails:** Multiple segments can have the same length (e.g., opposite sides of a rectangle). Using `length_to_id[length]` overwrites earlier entries, causing incorrect matches.

**Fallback for multi-entity constraints:** For constraints involving multiple entities (parallel, perpendicular), property-based matching (length, radius) can still be used as a fallback, but index-based lookup should be preferred when possible.

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

### 5a. Single-Entity Constraints (Horizontal, Vertical, Fixed)
For constraints that apply to a single entity, `GetEntities()` returns the constrained segment, but matching it back to our known segments fails due to the COM wrapper issue.

**Problem:**
```python
relation = segment.GetRelations[0]  # Horizontal constraint
entities = relation.GetEntities     # Returns the constrained segment
# But we can't match 'entities[0]' to our known segments!
```

**Solution:** For single-entity constraint types, the source segment we're iterating from IS the constrained entity. Use the source segment ID directly:
```python
single_entity_types = {HORIZONTAL, VERTICAL, FIXED}
if rel_type in single_entity_types:
    # Don't bother with GetEntities - use source segment
    refs = [source_segment_id]
else:
    # Multi-entity constraints - resolve via GetEntities
    refs = resolve_entities(relation.GetEntities)
```

### 5b. Constraint Deduplication
The same constraint may appear when iterating from multiple segments (e.g., connected segments both "see" constraints at their shared endpoint).

**Solution:** Track seen constraints by `(constraint_type, segment_id)` tuple:
```python
seen_relations = set()
dedup_key = (rel_type, segment_id)
if dedup_key in seen_relations:
    continue  # Skip duplicate
seen_relations.add(dedup_key)
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
2. **Use index-based segment mapping** for reliable entity identification (see §2)
3. **Use `FeatureByPositionReverse`** for feature iteration
4. **Access sketch via `feat.GetSpecificFeature2`** (uncalled)
5. **Iterate segments for relations** instead of calling sketch-level methods
6. **Use source segment ID for single-entity constraints** (horizontal, vertical, fixed)
7. **Deduplicate constraints by (type, segment_id)** to avoid duplicates
8. **Handle exceptions gracefully** - many operations fail silently in late-bound COM
