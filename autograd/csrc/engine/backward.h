#pragma once

#include <optional>

#include "engine/graph.h"


class AutogradEngine {
    public:
        static bool on_;
        static bool track_graph_;
        static ComputationalGraph graph_;

        static void on(bool flag);  // AutogradEngine::on setter
        static void track_graph(bool flag);  // AutogradEngine::track_graph setter
        static ComputationalGraph get_graph();
        static void clear_graph();

        static void backward(
            const TensorPtr& root,
            const std::optional<TensorPtr>& gradient = std::nullopt,
            const bool retain_graph = false
        );
};