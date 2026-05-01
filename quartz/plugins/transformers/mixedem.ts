import { QuartzTransformerPlugin } from "../types"
import { Root, Element, Text, ElementContent } from "hast"
import { visit } from "unist-util-visit"

function containsHan(text: string): boolean {
  return /\p{Script=Han}/u.test(text)
}

function splitByScript(text: string): ElementContent[] {
  const parts =
    text.match(
      /[\p{Script=Han}\u3000-\u303F\uFF00-\uFFEF]+|[^\p{Script=Han}\u3000-\u303F\uFF00-\uFFEF]+/gu,
    ) ?? []

  return parts.map((part): Element => {
    const isZh = containsHan(part)

    return {
      type: "element",
      tagName: "span",
      properties: {
        className: [isZh ? "em-zh" : "em-en"],
      },
      children: [
        {
          type: "text",
          value: part,
        } satisfies Text,
      ],
    }
  })
}

function transformEmphasisNode(node: Element) {
  // 只处理纯文本 emphasis，避免破坏链接、代码、脚注等复杂结构
  const onlyTextChildren = node.children.every((child) => child.type === "text")
  if (!onlyTextChildren) return

  const text = node.children
    .map((child) => (child.type === "text" ? child.value : ""))
    .join("")

  if (!text.trim()) return

  node.properties = {
    ...(node.properties ?? {}),
    className: [
      ...normalizeClassName(node.properties?.className),
      "mixed-emphasis",
    ],
  }

  node.children = splitByScript(text)
}

function normalizeClassName(className: unknown): string[] {
  if (Array.isArray(className)) return className.map(String)
  if (typeof className === "string") return className.split(/\s+/).filter(Boolean)
  return []
}

export const MixedEmphasis: QuartzTransformerPlugin = () => {
  return {
    name: "MixedEmphasis",

    htmlPlugins() {
      return [
        () => {
          return (tree: Root) => {
            visit(tree, "element", (node: Element) => {
              if (node.tagName !== "em") return
              transformEmphasisNode(node)
            })
          }
        },
      ]
    },
  }
}