import json

import attr
import torch
import networkx as nx

from seq2struct.utils import registry


@attr.s
class SpiderItem:
    text = attr.ib()
    code = attr.ib()
    schema = attr.ib()
    orig = attr.ib()
    orig_schema = attr.ib()


@attr.s
class Column:
    id = attr.ib()
    table = attr.ib()
    name = attr.ib()
    orig_name = attr.ib()
    type = attr.ib()
    foreign_key_for = attr.ib(default=None)


@attr.s
class Table:
    id = attr.ib()
    name = attr.ib()
    orig_name = attr.ib()
    columns = attr.ib(factory=list)
    primary_keys = attr.ib(factory=list)


@attr.s
class Schema:
    db_id = attr.ib()
    tables = attr.ib()
    columns = attr.ib()
    foreign_key_graph = attr.ib()


@registry.register('dataset', 'spider')
class SpiderDataset(torch.utils.data.Dataset): 
    def __init__(self, paths, tables_path, limit=None):
        self.paths = paths
        self.examples = []
        self.schemas = {}

        schema_dicts  = json.load(open(tables_path))
        for schema_dict in schema_dicts:
            tables = tuple(
                Table(id=i, name=name.split(), orig_name=orig_name)
                for i, (name, orig_name) in enumerate(zip(
                    schema_dict['table_names'], schema_dict['table_names_original']))
            )
            columns = tuple(
                Column(
                    id=i,
                    table=tables[table_id] if table_id >= 0 else None,
                    name=col_name.split(),
                    orig_name=orig_col_name,
                    type=col_type,
                )
                for i, ((table_id, col_name), (_, orig_col_name), col_type) in enumerate(zip(
                    schema_dict['column_names'], 
                    schema_dict['column_names_original'],
                    schema_dict['column_types']))
            )

            # Link columns to tables
            for column in columns:
                if column.table:
                    column.table.columns.append(column)
                    
            for column_id in schema_dict['primary_keys']:
                # Register primary keys
                column = columns[column_id]
                column.table.primary_keys.append(column)

            foreign_key_graph = nx.DiGraph()            
            for source_column_id, dest_column_id in schema_dict['foreign_keys']:
                # Register foreign keys
                source_column = columns[source_column_id]
                dest_column = columns[dest_column_id]
                source_column.foreign_key_for = dest_column
                foreign_key_graph.add_edge(
                    source_column.table.id,
                    dest_column.table.id,
                    columns=(source_column_id, dest_column_id))
                foreign_key_graph.add_edge(
                    dest_column.table.id,
                    source_column.table.id,
                    columns=(dest_column_id, source_column_id))

            db_id = schema_dict['db_id']
            self.schemas[db_id] = Schema(db_id, tables, columns, foreign_key_graph)

        schema_dicts_by_db = {s['db_id']: s for s in schema_dicts}

        for path in paths:
            raw_data = json.load(open(path))
            for entry in raw_data:
                item = SpiderItem(
                    text=entry['question_toks'],
                    code=entry['sql'],
                    schema=self.schemas[entry['db_id']],
                    orig=entry,
                    orig_schema=schema_dicts_by_db[entry['db_id']])
                self.examples.append(item)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
