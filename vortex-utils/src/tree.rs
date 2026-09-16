// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared traversal and rendering utilities for named trees.

use std::fmt;

/// Names the row that collects a node's empty child positions.
pub const EMPTY_NODE: &str = "<empty>";

/// Receives each named child position as a tree is rendered.
///
/// The node is `None` for a named position that holds no node.
pub type ChildVisitor<'a, N> = dyn FnMut(&str, Option<&N>) -> fmt::Result + 'a;

/// Traversal state updated as a tree renderer enters and leaves a node's children.
///
/// The context visible while a node is rendered describes its ancestors. The renderer calls
/// [`Self::push_parent`] before visiting the node's children and [`Self::pop_parent`] afterwards.
pub trait TreeDisplayContext<N: ?Sized> {
    /// Record `parent` while its children are visited.
    fn push_parent(&mut self, parent: &N) {
        _ = parent;
    }

    /// Remove `parent` after all of its children have been visited.
    fn pop_parent(&mut self, parent: &N) {
        _ = parent;
    }
}

impl<N: ?Sized> TreeDisplayContext<N> for () {}

/// Tree traversal context that records only the current depth.
#[derive(Debug, Default)]
pub struct DepthContext {
    depth: usize,
}

impl DepthContext {
    /// Return the current node's depth, where the root has depth zero.
    pub fn depth(&self) -> usize {
        self.depth
    }
}

impl<N: ?Sized> TreeDisplayContext<N> for DepthContext {
    fn push_parent(&mut self, _parent: &N) {
        self.depth += 1;
    }

    fn pop_parent(&mut self, _parent: &N) {
        debug_assert!(self.depth > 0, "tree depth push/pop mismatch");
        self.depth -= 1;
    }
}

/// Access to a formatter together with the indentation for detail lines.
pub struct IndentedFormatter<'a, 'b> {
    inner: &'a mut fmt::Formatter<'b>,
    indent: &'a str,
}

impl<'a, 'b> IndentedFormatter<'a, 'b> {
    fn new(inner: &'a mut fmt::Formatter<'b>, indent: &'a str) -> Self {
        Self { inner, indent }
    }

    /// Return the indentation string and underlying formatter together.
    pub fn parts(&mut self) -> (&str, &mut fmt::Formatter<'b>) {
        (self.indent, self.inner)
    }

    /// Return the current indentation string.
    pub fn indent(&self) -> &str {
        self.indent
    }

    /// Return the underlying formatter.
    pub fn formatter(&mut self) -> &mut fmt::Formatter<'b> {
        self.inner
    }
}

/// Contributes one composable dimension of information to tree nodes.
pub trait TreeDisplayExtractor<N: ?Sized, C: TreeDisplayContext<N>>: Send + Sync {
    /// Write space-prefixed annotations on the node's header line.
    fn write_header(
        &self,
        node: &N,
        context: &C,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        _ = (node, context, formatter);
        Ok(())
    }

    /// Write detail lines beneath the node's header.
    fn write_details(
        &self,
        node: &N,
        context: &C,
        formatter: &mut IndentedFormatter<'_, '_>,
    ) -> fmt::Result {
        _ = (node, context, formatter);
        Ok(())
    }
}

/// Adapts a domain-specific node and traversal context to the shared tree renderers.
pub trait TreeDisplayAdapter {
    /// Node type traversed by this adapter.
    type Node: ?Sized;

    /// State made available while each node is rendered.
    type Context: TreeDisplayContext<Self::Node>;

    /// Write the node's display content.
    ///
    /// The indented renderer writes `name:` first, so implementations normally write
    /// space-prefixed annotations. The branch renderer uses this as the complete node label.
    fn write_node(
        &self,
        node: &Self::Node,
        context: &Self::Context,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result;

    /// Write detail lines beneath an indented node header.
    fn write_details(
        &self,
        node: &Self::Node,
        context: &Self::Context,
        formatter: &mut IndentedFormatter<'_, '_>,
    ) -> fmt::Result {
        _ = (node, context, formatter);
        Ok(())
    }

    /// Visit each named child in display order.
    ///
    /// A `None` child is a named position that holds no node, such as an empty array slot.
    /// Renderers gather those names into a single trailing [`EMPTY_NODE`] row rather than
    /// descending into them, so adapters report positions in their natural order and leave
    /// placement to the renderer.
    ///
    /// Renderers call the visitor synchronously, and may call it more than once per rendered
    /// node, so adapters may pass either stored or temporarily owned nodes.
    fn visit_children(
        &self,
        node: &Self::Node,
        visit: &mut ChildVisitor<'_, Self::Node>,
    ) -> fmt::Result;
}

/// Count a node's occupied child positions and collect the names of its empty ones.
fn scan_children<A: TreeDisplayAdapter>(
    adapter: &A,
    node: &A::Node,
) -> Result<(usize, Vec<String>), fmt::Error> {
    let mut occupied = 0;
    let mut empty = Vec::new();
    adapter.visit_children(node, &mut |name, child| {
        match child {
            Some(_) => occupied += 1,
            None => empty.push(name.to_string()),
        }
        Ok(())
    })?;
    Ok((occupied, empty))
}

/// Render a named tree using two-space indentation.
///
/// Each node is written as `name:` followed by [`TreeDisplayAdapter::write_node`]. Detail lines
/// and child nodes are indented beneath it.
pub fn write_indented_tree<A: TreeDisplayAdapter>(
    adapter: &A,
    root_name: &str,
    root: &A::Node,
    context: &mut A::Context,
    formatter: &mut fmt::Formatter<'_>,
) -> fmt::Result {
    write_indented_node(adapter, root_name, root, context, "", formatter)
}

fn write_indented_node<A: TreeDisplayAdapter>(
    adapter: &A,
    name: &str,
    node: &A::Node,
    context: &mut A::Context,
    indent: &str,
    formatter: &mut fmt::Formatter<'_>,
) -> fmt::Result {
    write!(formatter, "{indent}{name}:")?;
    adapter.write_node(node, context, formatter)?;
    writeln!(formatter)?;

    let child_indent = format!("{indent}  ");
    {
        let mut indented = IndentedFormatter::new(formatter, &child_indent);
        adapter.write_details(node, context, &mut indented)?;
    }

    let (_, empty_names) = scan_children(adapter, node)?;

    context.push_parent(node);
    let result = adapter.visit_children(node, &mut |child_name, child| match child {
        Some(child) => write_indented_node(
            adapter,
            child_name,
            child,
            context,
            &child_indent,
            formatter,
        ),
        None => Ok(()),
    });
    context.pop_parent(node);
    result?;

    if !empty_names.is_empty() {
        writeln!(
            formatter,
            "{child_indent}{EMPTY_NODE}: {}",
            empty_names.join(", ")
        )?;
    }
    Ok(())
}

/// Render a tree using Unicode branch connectors.
///
/// The root contains only [`TreeDisplayAdapter::write_node`]. Descendants are prefixed with their
/// child name and connectors such as `├──` and `└──`. Detail lines are not rendered in this style.
pub fn write_branch_tree<A: TreeDisplayAdapter>(
    adapter: &A,
    root: &A::Node,
    context: &mut A::Context,
    formatter: &mut fmt::Formatter<'_>,
) -> fmt::Result {
    write_branch_node(adapter, root, context, "", formatter)
}

fn write_branch_node<A: TreeDisplayAdapter>(
    adapter: &A,
    node: &A::Node,
    context: &mut A::Context,
    prefix: &str,
    formatter: &mut fmt::Formatter<'_>,
) -> fmt::Result {
    adapter.write_node(node, context, formatter)?;

    let (occupied, empty_names) = scan_children(adapter, node)?;
    let mut visited = 0;

    context.push_parent(node);
    let result = adapter.visit_children(node, &mut |child_name, child| {
        let Some(child) = child else { return Ok(()) };
        visited += 1;
        let is_last = visited == occupied && empty_names.is_empty();
        writeln!(formatter)?;
        let connector = if is_last { "└── " } else { "├── " };
        write!(formatter, "{prefix}{connector}{child_name}: ")?;
        let child_prefix = format!("{prefix}{}", if is_last { "    " } else { "│   " });
        write_branch_node(adapter, child, context, &child_prefix, formatter)
    });
    context.pop_parent(node);
    result?;

    if !empty_names.is_empty() {
        writeln!(formatter)?;
        write!(
            formatter,
            "{prefix}└── {EMPTY_NODE}: {}",
            empty_names.join(", ")
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fmt;

    use super::ChildVisitor;
    use super::DepthContext;
    use super::TreeDisplayAdapter;
    use super::write_branch_tree;
    use super::write_indented_tree;

    struct TestNode {
        label: &'static str,
        children: Vec<(&'static str, Option<TestNode>)>,
    }

    struct TestAdapter;

    impl TreeDisplayAdapter for TestAdapter {
        type Context = DepthContext;
        type Node = TestNode;

        fn write_node(
            &self,
            node: &Self::Node,
            context: &Self::Context,
            formatter: &mut fmt::Formatter<'_>,
        ) -> fmt::Result {
            write!(formatter, "{}@{}", node.label, context.depth())
        }

        fn visit_children(
            &self,
            node: &Self::Node,
            visit: &mut ChildVisitor<'_, Self::Node>,
        ) -> fmt::Result {
            for (name, child) in &node.children {
                visit(name, child.as_ref())?;
            }
            Ok(())
        }
    }

    struct IndentedDisplay<'a>(&'a TestNode);

    impl fmt::Display for IndentedDisplay<'_> {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write_indented_tree(
                &TestAdapter,
                "root",
                self.0,
                &mut DepthContext::default(),
                formatter,
            )
        }
    }

    struct BranchDisplay<'a>(&'a TestNode);

    impl fmt::Display for BranchDisplay<'_> {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write_branch_tree(
                &TestAdapter,
                self.0,
                &mut DepthContext::default(),
                formatter,
            )
        }
    }

    fn tree() -> TestNode {
        TestNode {
            label: "parent",
            children: vec![
                (
                    "left",
                    Some(TestNode {
                        label: "branch",
                        children: vec![(
                            "leaf",
                            Some(TestNode {
                                label: "first",
                                children: Vec::new(),
                            }),
                        )],
                    }),
                ),
                ("middle", None),
                ("spare", None),
                (
                    "right",
                    Some(TestNode {
                        label: "second",
                        children: Vec::new(),
                    }),
                ),
            ],
        }
    }

    #[test]
    fn renders_indented_tree() {
        assert_eq!(
            IndentedDisplay(&tree()).to_string(),
            "root:parent@0\n  left:branch@1\n    leaf:first@2\n  right:second@1\n  <empty>: middle, spare\n"
        );
    }

    #[test]
    fn renders_branch_tree() {
        assert_eq!(
            BranchDisplay(&tree()).to_string(),
            "parent@0\n├── left: branch@1\n│   └── leaf: first@2\n├── right: second@1\n└── <empty>: middle, spare"
        );
    }
}
