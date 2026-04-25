# Sphinx Markdown Guide

This guide provides instructions for using Sphinx with Markdown to create technical documentation. It covers common scenarios such as nested directives, diagrams, figures, code inclusion, mathematical formulas, and table of contents.

## Nested Directives

In Markdown, Sphinx uses code blocks to define directives. However, markdown does not natively support nesting codeblocks within one another. To achieve nested directives, use triple tildes (`~~~`) as a replacement for triple backticks (```) for the second and subsequent nesting levels.

Example:

```{note}
   ~~~{note}
      ~~~{note}
        ~~~{important}
        Hello World!
        ~~~
      ~~~
   ~~~
```

In this example, each inner directive is enclosed with `~~~` instead of ``` to avoid conflicts with the outer code block delimiter.

## Diagrams

### Simple Diagrams with Mermaid

For simple flowcharts and diagrams, the Mermaid directive is recommended. It renders diagrams directly from textual descriptions.

Example:

```{mermaid}
---
name: test <optional>
alt: <alternate text for HTML output, optional>
align: left|center|right <optional>
caption: <caption text, optional>
---

graph LR
    A[Hard edge] -->|Link text| B(Round edge)
    B --> C{Decision}
    C -->|One| D[Result one]
    C -->|Two| E[Result two]
```

IF Mermaid cli is not available, the document will compile, but diagrams will not get rendered.

### Complex Diagrams with TikZ

For more complex diagrams requiring fine-grained control, TikZ can be used within a `tikz` directive.

Example:

```{tikz} <caption>
\begin{tikzpicture}
    \draw (0,0) -- (1,1) -- (2,0) -- (0,0);
\end{tikzpicture}
```

IF TexLive and pdf2svg is not available, the document will compile, but diagrams will not get rendered.

## Figures

To include an image with a caption, width, and alignment, use the `figure` directive. The path is relative to the Markdown file.

Example:

```{figure} images/example.png
:width: 400px
:align: center
:caption: An example figure illustrating the concept.
```

Remove the `:width:` or `:align:` lines if they are not required.

## Code Snippets

For large code blocks, it is recommended to store the code in a separate file and include it using the `literalinclude` directive. This improves maintainability and reduces clutter.

Example:

```{literalinclude} code/example.py
:language: python
:linenos:
:caption: Example Python code snippet.
```

To enable line numbering, omit the `:linenos:` line.

## Mathematical Formulas

Mathematical expressions can be written using the `math` directive or double‑dollar LaTeX delimiters.

Example with directive:

```{math}
E = mc^2
```

Example with LaTeX block:

$$
E = mc^2
$$

Both styles produce properly rendered equations.

## Coloured Box

To generate a coloured box to indicate the purpose or the nature of a paragraph, an `admonition` can be used.
Admonition have following classes: `attention`, `caution`, `danger`, `error`, `hint`, `important`, `note`, `tip` and `warning`

Example:

```{admonition}
:class: tip

This is a tip
```


## Table of Contents

To generate a table of contents within an admonition, combine the `admonition` and `toctree` directives. The `toctree` directive must be nested using tildes as described in the Nested Directives section.

Example:

```{admonition} Table of Contents
:class: note
~~~{toctree}
:maxdepth: 1

lesson2/index
lesson4/index
...
~~~
```

List the topics in the desired order. Each entry is the path of the Markdown file without the `.md` extension.

## Cross References

Sphinx provides powerful cross-referencing capabilities to link between different parts of your documentation.

### Referencing Sections with `{ref}`

To reference a specific section or heading, first define a label above the target header, then use `{ref}` to create the link.

**Step 1: Define a label** (must be above a header):

```markdown
(my-custom-label)=
# Target Section

Content here...
```

**Step 2: Reference the label** from any other file:

```markdown
See {ref}`my-custom-label` for more details.
```

Or with custom link text:

```markdown
See {ref}`Custom Link Text <my-custom-label>` for more details.
```

### Referencing Documents with `{doc}`

To link to the "front door" of another page (using its title automatically), use the `{doc}` role:

```markdown
Continue to {doc}`./folder/filename` for the next lesson.
```

Note: Omit the `.md` extension in the path.

### Referencing with Markdown Links

For external URLs or standard Markdown links:

```markdown
[Link text](https://example.com)
[Link to section](#section-name)
```

### Best Practices for Cross References

1. **Use descriptive labels**: Choose label names that clearly indicate the target content, e.g., `(gradient-descent)=` instead of `(section-5)=`.

2. **Define labels at the top of files**: Place `(label-name)=` right before the main `# Title` for easy reference.

3. **Reference early and often**: Link related concepts across chapters to help readers navigate.

4. **Use `{ref}` for sections, `{doc}` for files**: 
   - Use `{ref}` when pointing to a specific section/heading
   - Use `{doc}` when linking to an entire document

5. **Check references after restructuring**: If you move or rename files, update all cross references accordingly.

### Example Usage

```markdown
(computational-graph)=
# Computational Graphs

## Introduction
...

In the next section, we discuss {ref}`activation-functions`.
For a complete overview, see {doc}`index`.
```

## Conclusion

Following these patterns will ensure that your Sphinx‑based documentation is well‑structured, maintainable, and visually consistent. For further details, refer to the official [Sphinx documentation](https://www.sphinx-doc.org/).
