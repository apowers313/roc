## Features:

- Convenience: accessing edges from nodes or nodes from edges
  automatically retrieves them from the database and modifications to objects
  are automatically saved to the database. You don't have to think about
  when to load or save data.
- Performance: nodes and edges are cached locally, reducing database querries
  until they are needed.
- Reduce Errors: node and edge types are Pydantic models to strongly enforce
  data types and graph database schema is strongly defined through allowed
  connection types.

### Getting Started

Connecting to the database:

```python

```

Creating a node:

Creating an edge:

Connecting nodes:

Finding nodes:

Finding edges:

Updating a node:

Delete a node:

### Node and Edge Access

n -> e -> n

### Caching

Find node, find same node, they're the same object

### Schema

```python
class Person(Node):
    name: str

class Friend(Edge):
    allowed_connections = [("Person", "Person")]
```

```python
class Employee(Person):
    employee_id: int

class Manager(Person):
    tenure: int

class Manages(Edge):
    allowed_connections = [("Manager", "Employee")]
```

### Advanced

Successors
Predecessors
NodeList.select
EdgeList.select
GraphDB.to_networkx
Schema.to_mermaid
