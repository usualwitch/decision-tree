from anytree import Node, RenderTree

udo = Node("Udo")
marc = Node("Marc", parent=udo)
lian = Node("Lian", parent=marc)
dan = Node("Dan", parent=udo)
jet = Node("Jan", parent=udo)
joe = Node("Joe", parent=dan)
john = Node("John", parent=marc)

print(udo)

for pre, fill, node in RenderTree(udo):
    print("%s%s" % (pre, node.name))

from anytree.exporter import DotExporter
DotExporter(udo).to_picture("udo.png")
