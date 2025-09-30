import { QuartzConfig } from "./quartz/cfg"
import * as Plugin from "./quartz/plugins"

/**
 * Quartz 4.0 Configuration
 *
 * See https://quartz.jzhao.xyz/configuration for more information.
 */
const config: QuartzConfig = {
  configuration: {
    pageTitle: "🐱 Namoe",
    enableSPA: true,
    enablePopovers: true,
    analytics: null,
    locale: "en-US",
    baseUrl: "na-moe.github.io",
    ignorePatterns: ["private", "templates", ".obsidian"],
    defaultDateType: "modified",
    theme: {
      fontOrigin: "local",
      cdnCaching: false,
      typography: {
        header: "Alegreya SC",
        body: "Iowan-old-style",
        code: "Jetbrains Mono",
      },
      colors: {
        lightMode: {              // one half light
          light: "#fafafa",     // background
          lightgray: "#e3e2dd", // background - darkMode(black - background)
          gray: "#d3d0c8",      // background - darkMode(light black - background)
          darkgray: "#4f525d",  // black
          dark: "#383a42",      // light black
          secondary: "#0184bc", // blue
          tertiary: "#50a14f",  // green
          highlight: "#56b5c126",  // light cyan
          textHighlight: "#e4c07a88", // light yellow
        },
        darkMode: {               // one dark pro
          light: "#282c34",     // background
          lightgray: "#3f4451", // black
          gray: "#4f5666",      // light black
          darkgray: "#abb2bf",  // foreground
          dark: "#e6e6e6",      // light white
          secondary: "#61afef", // blue
          tertiary: "#98c379",  // green
          highlight: "#4cd1e026",  // light cyan
          textHighlight: "#e5c07b88", // light yellow
        },
      },
    },
  },
  plugins: {
    transformers: [
      Plugin.FrontMatter(),
      Plugin.CreatedModifiedDate({
        priority: ["frontmatter", "git", "filesystem"],
      }),
      Plugin.SyntaxHighlighting({
        theme: {
          light: "one-light",
          dark: "one-dark-pro",
        },
        keepBackground: false,
      }),
      Plugin.ObsidianFlavoredMarkdown({ enableInHtmlEmbed: false }),
      Plugin.GitHubFlavoredMarkdown(),
      Plugin.TableOfContents(),
      Plugin.CrawlLinks({ markdownLinkResolution: "shortest" }),
      Plugin.Description(),
      Plugin.Latex({ renderEngine: "katex" }),
    ],
    filters: [Plugin.RemoveDrafts()],
    emitters: [
      Plugin.AliasRedirects(),
      Plugin.ComponentResources(),
      Plugin.ContentPage(),
      Plugin.FolderPage(),
      Plugin.TagPage(),
      Plugin.ContentIndex({
        enableSiteMap: true,
        enableRSS: true,
      }),
      Plugin.Assets(),
      Plugin.Static(),
      Plugin.Favicon(),
      Plugin.NotFoundPage(),
      // Comment out CustomOgImages to speed up build time
      Plugin.CustomOgImages(),
    ],
  },
}

export default config
