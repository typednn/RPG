# TypedNN: a domain-specific language for configuring and training neural networks  

The goal of the language is to simplify the coding of neural networks that takes different types of input. It has the following features

1. We support an augmented [HM type system](https://en.wikipedia.org/wiki/Hindley%E2%80%93Milner_type_system), enabling simple shape inference and type check.
2. Procedure-based module definition: one can define new operators using Python function directly.
3. Object-oriented: we enable inheritance and class methods so that we can write true object-oriented code. 
4. All components of the computation graph can be configured with external configs, which supports the research purpose naturally. 

In the end, we hope that building a deep learning algorithm can be as simple as writing math formulas.


Now let's state the key elements of the language:

- A piece of a program describes a computation graph, formed by operators and computation nodes. Each node is associated with two types: the meta type and the actual type. The meta type has to be decided before initializing the modules, at least determining if a node is a dict, tuple or a type that has certain attributes, but it usually does not have to worry about the actual dimensions of the tensor shapes or actual value of the primitive types. The true type will be determined after initializing the operators in the end. 
- Operator: can be viewed as parameterized function of datas. An operator (CallNode) will be created after feeding computation nodes into a MetaOperator (e.g., the classname of the operator) and it will map inputs to create a new node and an instance of the operator.
  - an operator can reuse (actually by default it will) parameters for different input types if it is configured well.
- Function: a function composes operators into new a new MetaOperator (i.e., operator class). One can reuse this function to create new operators.
- AttrType, or classes: For any subclasses of the AttrType will support class method. Nodes of these types are able to call methods as operators. One can specify some rules to determine which methods to use if necessary.

----

It has several benefits:
- Unified interface for different types of input, e.g., text, image, audio, video, etc. Providing ways of automatically generating the neural networks based on the type of inputs.
- unified config and log system.
- High-level abstractions like ``partial''/condition and computation graphs for probabilistic programming -> support functor types!
- We provide strict type check and shape inference for the neural networks, with better traceback information, which can help us to find bugs early.
- Interface-like object-oriented design. 
- With the help of automatical type inference, we only need to define the networks once, e.g., we don't have to define/config the module and write the forward separately.