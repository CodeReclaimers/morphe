#include "morphe/serialization.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <variant>

#include "nlohmann/json.hpp"

namespace morphe {

// We use nlohmann::ordered_json for output to preserve key insertion order
// (the default nlohmann::json sorts keys alphabetically because it uses
// std::map under the hood). This is how we match Python's dict-iteration
// order in the wire format.
using ojson = nlohmann::ordered_json;
using ijson = nlohmann::json;  // input parsing; key order does not matter here

namespace {

// ---------- helpers ----------

ojson point2d_to_array(const Point2D& p) {
    return ojson::array({p.x, p.y});
}

Point2D parse_point(const ijson& a) {
    // Mirrors morphe.serialization._parse_point: tolerate missing components,
    // default to 0.0.
    if (!a.is_array()) return Point2D{};
    Point2D out{};
    if (a.size() >= 1 && a[0].is_number()) out.x = a[0].get<double>();
    if (a.size() >= 2 && a[1].is_number()) out.y = a[1].get<double>();
    return out;
}

ojson point_ref_to_json(const PointRef& r) {
    ojson out = ojson::object();
    out["element"] = r.element_id;
    out["point"]   = std::string{to_string(r.point_type)};
    if (r.parameter.has_value()) out["parameter"] = *r.parameter;
    if (r.index.has_value())     out["index"]     = *r.index;
    return out;
}

PointRef point_ref_from_json(const ijson& j) {
    PointRef ref;
    if (j.contains("element"))  ref.element_id = j.at("element").get<std::string>();
    if (j.contains("point"))    ref.point_type = point_type_from_string(j.at("point").get<std::string>());
    else                        ref.point_type = PointType::Center;  // matches Python default
    if (j.contains("parameter") && !j.at("parameter").is_null())
        ref.parameter = j.at("parameter").get<double>();
    if (j.contains("index") && !j.at("index").is_null())
        ref.index = j.at("index").get<int>();
    return ref;
}

// ---------- primitives ----------

void write_meta(ojson& out, const PrimitiveMeta& meta, std::string_view type_str) {
    out["id"]           = meta.id;
    out["type"]         = std::string{type_str};
    out["construction"] = meta.construction;
    if (meta.source.has_value())   out["source"]     = *meta.source;
    if (meta.confidence != 1.0)    out["confidence"] = meta.confidence;
}

ojson primitive_to_json(const Primitive& p) {
    ojson out = ojson::object();
    write_meta(out, meta_of(p), type_tag(p));

    struct Visitor {
        ojson& out;
        void operator()(const Line& l) const {
            out["start"] = point2d_to_array(l.start);
            out["end"]   = point2d_to_array(l.end);
        }
        void operator()(const Arc& a) const {
            out["center"]      = point2d_to_array(a.center);
            out["start_point"] = point2d_to_array(a.start_point);
            out["end_point"]   = point2d_to_array(a.end_point);
            out["ccw"]         = a.ccw;
        }
        void operator()(const Circle& c) const {
            out["center"] = point2d_to_array(c.center);
            out["radius"] = c.radius;
        }
        void operator()(const Point& p) const {
            out["position"] = point2d_to_array(p.position);
        }
        void operator()(const Spline& s) const {
            out["degree"] = s.degree;
            ojson cps = ojson::array();
            for (const auto& cp : s.control_points) cps.push_back(point2d_to_array(cp));
            out["control_points"] = std::move(cps);
            ojson knots = ojson::array();
            for (double k : s.knots) knots.push_back(k);
            out["knots"]         = std::move(knots);
            out["periodic"]      = s.periodic;
            out["is_fit_spline"] = s.is_fit_spline;
            if (s.weights.has_value()) {
                ojson w = ojson::array();
                for (double v : *s.weights) w.push_back(v);
                out["weights"] = std::move(w);
            }
        }
        void operator()(const Ellipse& e) const {
            out["center"]       = point2d_to_array(e.center);
            out["major_radius"] = e.major_radius;
            out["minor_radius"] = e.minor_radius;
            out["rotation"]     = e.rotation;
        }
        void operator()(const EllipticalArc& e) const {
            out["center"]       = point2d_to_array(e.center);
            out["major_radius"] = e.major_radius;
            out["minor_radius"] = e.minor_radius;
            out["rotation"]     = e.rotation;
            out["start_param"]  = e.start_param;
            out["end_param"]    = e.end_param;
            out["ccw"]          = e.ccw;
        }
    };
    std::visit(Visitor{out}, p);
    return out;
}

void read_meta(PrimitiveMeta& meta, const ijson& j) {
    if (j.contains("id"))           meta.id           = j.at("id").get<std::string>();
    if (j.contains("construction")) meta.construction = j.at("construction").get<bool>();
    if (j.contains("source") && !j.at("source").is_null())
        meta.source = j.at("source").get<std::string>();
    if (j.contains("confidence") && !j.at("confidence").is_null())
        meta.confidence = j.at("confidence").get<double>();
}

Primitive primitive_from_json(const ijson& j) {
    const auto type_str = j.value("type", std::string{});

    if (type_str == "line") {
        Line out;
        read_meta(out.meta, j);
        if (j.contains("start")) out.start = parse_point(j.at("start"));
        if (j.contains("end"))   out.end   = parse_point(j.at("end"));
        return out;
    }
    if (type_str == "arc") {
        Arc out;
        read_meta(out.meta, j);
        if (j.contains("center"))      out.center      = parse_point(j.at("center"));
        if (j.contains("start_point")) out.start_point = parse_point(j.at("start_point"));
        if (j.contains("end_point"))   out.end_point   = parse_point(j.at("end_point"));
        if (j.contains("ccw"))         out.ccw         = j.at("ccw").get<bool>();
        return out;
    }
    if (type_str == "circle") {
        Circle out;
        read_meta(out.meta, j);
        if (j.contains("center")) out.center = parse_point(j.at("center"));
        if (j.contains("radius")) out.radius = j.at("radius").get<double>();
        return out;
    }
    if (type_str == "point") {
        Point out;
        read_meta(out.meta, j);
        if (j.contains("position")) out.position = parse_point(j.at("position"));
        return out;
    }
    if (type_str == "spline") {
        Spline out;
        read_meta(out.meta, j);
        if (j.contains("degree")) out.degree = j.at("degree").get<int>();
        if (j.contains("control_points")) {
            for (const auto& cp : j.at("control_points")) {
                out.control_points.push_back(parse_point(cp));
            }
        }
        if (j.contains("knots")) {
            for (const auto& k : j.at("knots")) {
                out.knots.push_back(k.get<double>());
            }
        }
        if (j.contains("weights") && !j.at("weights").is_null()) {
            std::vector<double> w;
            for (const auto& v : j.at("weights")) w.push_back(v.get<double>());
            out.weights = std::move(w);
        }
        if (j.contains("periodic"))      out.periodic      = j.at("periodic").get<bool>();
        if (j.contains("is_fit_spline")) out.is_fit_spline = j.at("is_fit_spline").get<bool>();
        return out;
    }
    if (type_str == "ellipse") {
        Ellipse out;
        read_meta(out.meta, j);
        if (j.contains("center"))       out.center       = parse_point(j.at("center"));
        if (j.contains("major_radius")) out.major_radius = j.at("major_radius").get<double>();
        if (j.contains("minor_radius")) out.minor_radius = j.at("minor_radius").get<double>();
        if (j.contains("rotation"))     out.rotation     = j.at("rotation").get<double>();
        return out;
    }
    if (type_str == "ellipticalarc") {
        EllipticalArc out;
        read_meta(out.meta, j);
        if (j.contains("center"))       out.center       = parse_point(j.at("center"));
        if (j.contains("major_radius")) out.major_radius = j.at("major_radius").get<double>();
        if (j.contains("minor_radius")) out.minor_radius = j.at("minor_radius").get<double>();
        if (j.contains("rotation"))     out.rotation     = j.at("rotation").get<double>();
        if (j.contains("start_param"))  out.start_param  = j.at("start_param").get<double>();
        if (j.contains("end_param"))    out.end_param    = j.at("end_param").get<double>();
        if (j.contains("ccw"))          out.ccw          = j.at("ccw").get<bool>();
        return out;
    }
    throw std::invalid_argument("unknown primitive type: " + type_str);
}

// ---------- constraints ----------

ojson constraint_to_json(const SketchConstraint& c) {
    ojson out = ojson::object();
    out["id"]   = c.id;
    out["type"] = std::string{to_string(c.constraint_type)};

    ojson refs = ojson::array();
    for (const auto& r : c.references) {
        std::visit([&](const auto& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, std::string>) {
                refs.push_back(v);
            } else {
                refs.push_back(point_ref_to_json(v));
            }
        }, r);
    }
    out["references"] = std::move(refs);

    if (c.value.has_value())            out["value"]            = *c.value;
    if (c.connection_point.has_value()) out["connection_point"] = point_ref_to_json(*c.connection_point);
    if (c.inferred)                     out["inferred"]         = c.inferred;
    if (c.confidence != 1.0)            out["confidence"]       = c.confidence;
    if (c.source.has_value())           out["source"]           = *c.source;
    if (c.status != ConstraintStatus::Unknown)
        out["status"] = std::string{to_string(c.status)};
    return out;
}

SketchConstraint constraint_from_json(const ijson& j) {
    SketchConstraint c;
    if (j.contains("id"))   c.id              = j.at("id").get<std::string>();
    if (j.contains("type")) c.constraint_type = constraint_type_from_string(j.at("type").get<std::string>());

    if (j.contains("references")) {
        for (const auto& r : j.at("references")) {
            if (r.is_string()) {
                c.references.emplace_back(r.get<std::string>());
            } else if (r.is_object()) {
                c.references.emplace_back(point_ref_from_json(r));
            } else {
                throw std::invalid_argument("constraint reference must be string or object");
            }
        }
    }
    if (j.contains("value") && !j.at("value").is_null())
        c.value = j.at("value").get<double>();
    if (j.contains("connection_point") && !j.at("connection_point").is_null())
        c.connection_point = point_ref_from_json(j.at("connection_point"));
    if (j.contains("inferred"))   c.inferred   = j.at("inferred").get<bool>();
    if (j.contains("confidence")) c.confidence = j.at("confidence").get<double>();
    if (j.contains("source") && !j.at("source").is_null())
        c.source = j.at("source").get<std::string>();
    if (j.contains("status"))
        c.status = constraint_status_from_string(j.at("status").get<std::string>());
    return c;
}

}  // namespace

// ---------- public API ----------

std::string to_json(const SketchDocument& doc, int indent) {
    ojson out = ojson::object();
    out["name"] = doc.name;

    ojson prims = ojson::array();
    for (const auto& p : doc.primitives) prims.push_back(primitive_to_json(p));
    out["primitives"] = std::move(prims);

    ojson constraints = ojson::array();
    for (const auto& c : doc.constraints) constraints.push_back(constraint_to_json(c));
    out["constraints"] = std::move(constraints);

    out["solver_status"]       = std::string{to_string(doc.solver_status)};
    out["degrees_of_freedom"]  = doc.degrees_of_freedom;

    return out.dump(indent);
}

SketchDocument from_json(std::string_view json_text) {
    const ijson j = ijson::parse(json_text);
    SketchDocument doc;

    if (j.contains("name"))               doc.name = j.at("name").get<std::string>();
    else                                  doc.name = "Untitled";

    if (j.contains("primitives")) {
        for (const auto& p : j.at("primitives")) {
            Primitive prim = primitive_from_json(p);
            const std::string id = meta_of(prim).id;
            doc.add_primitive_with_id(std::move(prim), id);
        }
    }

    if (j.contains("constraints")) {
        for (const auto& c : j.at("constraints")) {
            doc.constraints.push_back(constraint_from_json(c));
        }
    }

    if (j.contains("solver_status"))
        doc.solver_status = solver_status_from_string(j.at("solver_status").get<std::string>());
    else
        doc.solver_status = SolverStatus::Dirty;

    if (j.contains("degrees_of_freedom"))
        doc.degrees_of_freedom = j.at("degrees_of_freedom").get<int>();

    return doc;
}

void save(const SketchDocument& doc, const std::filesystem::path& path, int indent) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot open for write: " + path.string());
    out << to_json(doc, indent);
}

SketchDocument load(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open for read: " + path.string());
    std::ostringstream buf;
    buf << in.rdbuf();
    return from_json(buf.str());
}

}  // namespace morphe
