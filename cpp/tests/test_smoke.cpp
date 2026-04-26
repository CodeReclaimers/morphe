#include "doctest/doctest.h"

#include "morphe/morphe.hpp"
#include "nlohmann/json.hpp"

TEST_CASE("morphe version constants are populated") {
    CHECK(morphe::version_major == 0);
    CHECK(morphe::version_minor == 1);
    CHECK(morphe::version_patch == 0);
}

TEST_CASE("nlohmann::json round-trips a trivial document") {
    nlohmann::json j;
    j["name"] = "smoke";
    j["primitives"] = nlohmann::json::array();
    const auto s = j.dump();
    const auto k = nlohmann::json::parse(s);
    CHECK(k["name"].get<std::string>() == "smoke");
    CHECK(k["primitives"].is_array());
    CHECK(k["primitives"].empty());
}
