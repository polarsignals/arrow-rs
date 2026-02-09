// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#[macro_use]
extern crate criterion;

use criterion::Criterion;
use std::ops::Range;

use rand::Rng;

extern crate arrow;

use arrow::datatypes::*;
use arrow::util::test_util::seedable_rng;
use arrow::{array::*, util::bench_util::*};
use arrow_array::builder::{
    BooleanBufferBuilder, GenericListViewBuilder, Int32Builder, Int64Builder,
    StringDictionaryBuilder,
};
use arrow_array::types::Int64Type;
use arrow_array::OffsetSizeTrait;
use arrow_buffer::ScalarBuffer;
use arrow_select::interleave::interleave;
use std::hint;
use std::sync::Arc;

fn do_bench(
    c: &mut Criterion,
    prefix: &str,
    len: usize,
    base: &dyn Array,
    slices: &[Range<usize>],
) {
    let arrays: Vec<_> = slices
        .iter()
        .map(|r| base.slice(r.start, r.end - r.start))
        .collect();
    let values: Vec<_> = arrays.iter().map(|x| x.as_ref()).collect();
    bench_values(
        c,
        &format!("interleave {prefix} {len} {slices:?}"),
        len,
        &values,
    );
}

fn bench_values(c: &mut Criterion, name: &str, len: usize, values: &[&dyn Array]) {
    let mut rng = seedable_rng();
    let indices: Vec<_> = (0..len)
        .map(|_| {
            let array_idx = rng.random_range(0..values.len());
            let value_idx = rng.random_range(0..values[array_idx].len());
            (array_idx, value_idx)
        })
        .collect();

    c.bench_function(name, |b| {
        b.iter(|| hint::black_box(interleave(values, &indices).unwrap()))
    });
}

fn create_list_view_array<O: OffsetSizeTrait>(
    size: usize,
    null_density: f32,
    list_len: usize,
) -> GenericListViewArray<O> {
    let mut rng = seedable_rng();
    let mut builder = GenericListViewBuilder::<O, _>::new(Int64Builder::new());
    for _ in 0..size {
        if rng.random::<f32>() < null_density {
            builder.append(false);
        } else {
            for _ in 0..list_len {
                if rng.random::<f32>() < null_density {
                    builder.values().append_null();
                } else {
                    builder.values().append_value(rng.random::<i64>());
                }
            }
            builder.append(true);
        }
    }
    builder.finish()
}

/// Creates a ListView<Struct<dict_str, i32>> array.
/// Each list element is a struct with a dictionary-encoded string field and an i32 field.
fn create_list_view_struct_dict_array(
    size: usize,
    null_density: f32,
    list_len: usize,
) -> ListViewArray {
    let mut rng = seedable_rng();

    // Build the child struct arrays: total child length = size * list_len (approx)
    let mut dict_builder = StringDictionaryBuilder::<Int32Type>::new();
    let mut int_builder = Int32Builder::new();
    let dict_values = ["alpha", "beta", "gamma", "delta", "epsilon"];

    let mut offsets = Vec::with_capacity(size);
    let mut sizes = Vec::with_capacity(size);
    let mut null_buf = BooleanBufferBuilder::new(size);
    let mut child_len = 0usize;

    for _ in 0..size {
        if rng.random::<f32>() < null_density {
            offsets.push(child_len as i32);
            sizes.push(0i32);
            null_buf.append(false);
        } else {
            offsets.push(child_len as i32);
            sizes.push(list_len as i32);
            null_buf.append(true);
            for _ in 0..list_len {
                let v = dict_values[rng.random_range(0..dict_values.len())];
                if rng.random::<f32>() < null_density {
                    dict_builder.append_null();
                } else {
                    dict_builder.append_value(v);
                }
                if rng.random::<f32>() < null_density {
                    int_builder.append_null();
                } else {
                    int_builder.append_value(rng.random::<i32>());
                }
            }
            child_len += list_len;
        }
    }

    let dict_array = Arc::new(dict_builder.finish()) as ArrayRef;
    let int_array = Arc::new(int_builder.finish()) as ArrayRef;

    let struct_fields = Fields::from(vec![
        Field::new("d", dict_array.data_type().clone(), true),
        Field::new("i", DataType::Int32, true),
    ]);
    let struct_array = StructArray::new(struct_fields.clone(), vec![dict_array, int_array], None);

    let field = Arc::new(Field::new_struct("item", struct_fields, true));
    ListViewArray::new(
        field,
        ScalarBuffer::from(offsets),
        ScalarBuffer::from(sizes),
        Arc::new(struct_array),
        Some(null_buf.finish().into()),
    )
}

fn add_benchmark(c: &mut Criterion) {
    let i32 = create_primitive_array::<Int32Type>(1024, 0.);
    let i32_opt = create_primitive_array::<Int32Type>(1024, 0.5);
    let string = create_string_array_with_len::<i32>(1024, 0., 20);
    let string_opt = create_string_array_with_len::<i32>(1024, 0.5, 20);
    let values = create_string_array_with_len::<i32>(10, 0.0, 20);
    let dict = create_dict_from_values::<Int32Type>(1024, 0.0, &values);

    let struct_i32_no_nulls_i32_no_nulls = StructArray::new(
        Fields::from(vec![
            Field::new("a", Int32Type::DATA_TYPE, false),
            Field::new("b", Int32Type::DATA_TYPE, false),
        ]),
        vec![
            Arc::new(create_primitive_array::<Int32Type>(1024, 0.)),
            Arc::new(create_primitive_array::<Int32Type>(1024, 0.)),
        ],
        None,
    );

    let struct_string_no_nulls_string_no_nulls = StructArray::new(
        Fields::from(vec![
            Field::new("a", DataType::Utf8, false),
            Field::new("b", DataType::Utf8, false),
        ]),
        vec![
            Arc::new(create_string_array_with_len::<i32>(1024, 0., 20)),
            Arc::new(create_string_array_with_len::<i32>(1024, 0., 20)),
        ],
        None,
    );

    let struct_i32_no_nulls_string_no_nulls = StructArray::new(
        Fields::from(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Utf8, false),
        ]),
        vec![
            Arc::new(create_primitive_array::<Int32Type>(1024, 0.)),
            Arc::new(create_string_array_with_len::<i32>(1024, 0., 20)),
        ],
        None,
    );

    let values = create_string_array_with_len::<i32>(1024, 0.0, 20);
    let sparse_dict = create_sparse_dict_from_values::<Int32Type>(1024, 0.0, &values, 10..20);

    let string_view = create_string_view_array(1024, 0.0);

    // use 8192 as a standard list size for better coverage
    let list_i64 = create_primitive_list_array_with_seed::<i32, Int64Type>(8192, 0.1, 0.1, 20, 42);
    let list_i64_no_nulls =
        create_primitive_list_array_with_seed::<i32, Int64Type>(8192, 0.0, 0.0, 20, 42);
    let list_view_i64 = create_list_view_array::<i32>(1024, 0., 20);
    let list_view_i64_opt = create_list_view_array::<i32>(1024, 0.1, 20);
    let list_view_struct_dict = create_list_view_struct_dict_array(1024, 0.0, 20);
    let list_view_struct_dict_opt = create_list_view_struct_dict_array(1024, 0.1, 20);

    let cases: &[(&str, &dyn Array)] = &[
        ("i32(0.0)", &i32),
        ("i32(0.5)", &i32_opt),
        ("str(20, 0.0)", &string),
        ("str(20, 0.5)", &string_opt),
        ("dict(20, 0.0)", &dict),
        ("dict_sparse(20, 0.0)", &sparse_dict),
        ("str_view(0.0)", &string_view),
        (
            "struct(i32(0.0), i32(0.0)",
            &struct_i32_no_nulls_i32_no_nulls,
        ),
        (
            "struct(str(20, 0.0), str(20, 0.0))",
            &struct_string_no_nulls_string_no_nulls,
        ),
        (
            "struct(i32(0.0), str(20, 0.0)",
            &struct_i32_no_nulls_string_no_nulls,
        ),
        ("list<i64>(0.1,0.1,20)", &list_i64),
        ("list<i64>(0.0,0.0,20)", &list_i64_no_nulls),
        ("list_view<i64>(0.0,0.0,20)", &list_view_i64),
        ("list_view<i64>(0.1,0.1,20)", &list_view_i64_opt),
        ("list_view<struct(dict,i32)>(0.0,20)", &list_view_struct_dict),
        ("list_view<struct(dict,i32)>(0.1,20)", &list_view_struct_dict_opt),
    ];

    for (prefix, base) in cases {
        let slices: &[(usize, &[_])] = &[
            (100, &[0..100, 100..230, 450..1000]),
            (400, &[0..100, 100..230, 450..1000]),
            (1024, &[0..100, 100..230, 450..1000]),
            (1024, &[0..100, 100..230, 450..1000, 0..1000]),
        ];

        for (len, slice) in slices {
            do_bench(c, prefix, *len, *base, slice);
        }
    }

    for len in [100, 1024, 2048] {
        bench_values(
            c,
            &format!("interleave dict_distinct {len}"),
            100,
            &[&dict, &sparse_dict],
        );
    }
}

criterion_group!(benches, add_benchmark);
criterion_main!(benches);
