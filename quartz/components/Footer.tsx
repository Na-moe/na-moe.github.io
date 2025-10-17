import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import style from "./styles/footer.scss"
import { version } from "../../package.json"
import { i18n } from "../i18n"

interface Options {
  links: Record<string, string>
}

export default ((opts?: Options) => {
  const Footer: QuartzComponent = ({ displayClass, cfg }: QuartzComponentProps) => {
    const year = new Date().getFullYear()
    const links = opts?.links ?? []
    return (
      <footer class={`${displayClass ?? ""}`}>
        {/* <p>
          {i18n(cfg.locale).components.footer.createdWith}{" "}
          <a href="https://quartz.jzhao.xyz/">Quartz v{version}</a> © {year}
        </p> */}
        <ul>
          {Object.entries(links).map(([text, link]) => {
            const GitHubImg = <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" viewBox="0 0 24 24"><path fill="currentColor" d="M12 2A10 10 0 0 0 2 12c0 4.42 2.87 8.17 6.84 9.5c.5.08.66-.23.66-.5v-1.69c-2.77.6-3.36-1.34-3.36-1.34c-.46-1.16-1.11-1.47-1.11-1.47c-.91-.62.07-.6.07-.6c1 .07 1.53 1.03 1.53 1.03c.87 1.52 2.34 1.07 2.91.83c.09-.65.35-1.09.63-1.34c-2.22-.25-4.55-1.11-4.55-4.92c0-1.11.38-2 1.03-2.71c-.1-.25-.45-1.29.1-2.64c0 0 .84-.27 2.75 1.02c.79-.22 1.65-.33 2.5-.33s1.71.11 2.5.33c1.91-1.29 2.75-1.02 2.75-1.02c.55 1.35.2 2.39.1 2.64c.65.71 1.03 1.6 1.03 2.71c0 3.82-2.34 4.66-4.57 4.91c.36.31.69.92.69 1.85V21c0 .27.16.59.67.5C19.14 20.16 22 16.42 22 12A10 10 0 0 0 12 2"/></svg>

            const GoogleScholarImg = <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" viewBox="0 0 24 24"><path fill="currentColor" d="M22.75 8.5v5a.75.75 0 0 1-1.5 0v-2.469l-.023.011l-1.477.693v4.89c0 1.909-1.527 3.193-3.05 3.953c-1.564.78-3.428 1.172-4.7 1.172s-3.136-.392-4.7-1.172c-1.523-.76-3.05-2.044-3.05-3.953v-4.89l-1.477-.692C1.721 10.549 1.25 9.478 1.25 8.5s.47-2.05 1.523-2.542L9.464 2.82a5.92 5.92 0 0 1 5.072 0l6.69 3.137C22.28 6.45 22.75 7.522 22.75 8.5m-8.214 5.68a5.92 5.92 0 0 1-5.072 0L5.75 12.437v4.187c0 1.01.82 1.913 2.22 2.61c1.36.679 2.996 1.015 4.03 1.015s2.67-.336 4.03-1.014c1.4-.698 2.22-1.601 2.22-2.61v-4.188z"/></svg>
            
            const RSSImg = <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" viewBox="0 0 24 24" ><path fill="currentColor" d="M5 21q-.825 0-1.412-.587T3 19t.588-1.412T5 17t1.413.588T7 19t-.587 1.413T5 21m12 0q0-2.925-1.1-5.462t-3-4.438t-4.437-3T3 7V4q3.55 0 6.625 1.325t5.4 3.65t3.65 5.4T20 21zm-6 0q0-1.675-.625-3.113T8.65 15.35t-2.537-1.725T3 13v-3q2.3 0 4.288.863t3.487 2.362t2.363 3.488T14 21z"/></svg>

            const isGitHub = text === "GitHub";
            const isGoogleScholar = text === "Google Scholar";
            const isRSS = text === "RSS";

            const content = isGitHub? GitHubImg : (isGoogleScholar? GoogleScholarImg : (isRSS? RSSImg : text));
            return (
              <li>
                <a href={link}>{content}</a>
              </li>
            )
          })}
        </ul>
      </footer>
    )
  }

  Footer.css = style
  return Footer
}) satisfies QuartzComponentConstructor
