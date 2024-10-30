enum layer_type {
    LAYER_TYPE_SIGMOID,
};

struct layer {
    layer_type type;
    void *data;
};

struct network {
    layer *layers;
    size_t layer_count;
};
