from docutils import nodes
from sphinx.util.docutils import SphinxDirective
from sphinx.util.parsing import nested_parse_to_nodes
from importlib import import_module

def resolve_attr(module, dotted_name):
    obj = module
    for part in dotted_name.split("."):
        obj = getattr(obj, part)
    return obj

class Dargs(SphinxDirective):
    required_arguments = 0
    optional_arguments = 0
    option_spec = {
        "module": str,
        "func": str,
    }

    def run(self):
        module_name = self.options.get("module")
        func_name = self.options.get("func")

        if not module_name or not func_name:
            error = self.state_machine.reporter.error(
                'Both "module" and "func" options are required.',
                nodes.literal_block(self.block_text, self.block_text),
                line=self.lineno,
            )
            return [error]

        module = import_module(module_name)
        func = resolve_attr(module, func_name)
        result = func()
        content = []
        if not isinstance(result, list):
            result = [result]
        for arg in result:
            content.append(arg.gen_doc(make_anchor=True, make_link=True))
        docs = "\n\n".join(content)
        node = nested_parse_to_nodes(self.state, docs, source=docs)
        return node


def setup(app):
    app.add_directive("dargs", Dargs)
