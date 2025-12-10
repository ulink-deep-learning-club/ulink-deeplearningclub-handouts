when using nested directives:

```{note}
   ~~~{note}
      ~~~{note}
        ~~~{important}
        Hallo World!
        ~~~
      ~~~
   ~~~
```

sphinx uses code blocks to use directives, so you can use code blocks to write nested directives.
but it does not support directive nesting. so you should use ~~~ as a replacement for ``` for level 2 (and higher nesting levels) directives

when drawing simple diagrams, mermaid is recommended:

```{mermaid}
graph LR
    A[Hard edge] -->|Link text| B(Round edge)
    B --> C{Decision}
    C -->|One| D[Result one]
    C -->|Two| E[Result two]
```

when drawing complex diagrams, tikz is recommended:

```{tikz}
\begin{tikzpicture}
    \draw (0,0) -- (1,1) -- (2,0) -- (0,0);
\end{tikzpicture}
```

when including a figure:

```{figure} <path relative to the md file>
:width: <the width of the figure, remove this line if the width is not needed>
:align: <the alignment of the figure, remove this line if the alignment is not needed>
:caption: <the caption of the figure>
```

when writing large code snippets, it is recocommended to write them in a separate file, then include them with:

```{literalinclude} <path relative to the md file>
:language: <programming language>
:linenos: <disable the line numbering, remove this line if line numbering is needed>
:caption: <the caption of the code block>
```

when writing math formulas, it is recommended to use the following syntax:

```{math}
E = mc^2
```

or 

$$
E = mc^2
$$


when writing toc:

```{admonition} 目录
:class: note
~~~{toctree}
:maxdepth: 1

lesson2/index
lesson4/index
...
<list the topics here in order, the topic paths are the paths of the md files you want to include without the extension name>
~~~
```