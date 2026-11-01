#ifndef LAUNCHER_INCLUDE_FORK_INFO_H_
#define LAUNCHER_INCLUDE_FORK_INFO_H_

// The Yōtei fork's own version and addresses (sidebar footer, About, bug report, update check).

constexpr char FORK_VERSION[] = "v0.1";

// The fork's GitHub repository, for example "https://github.com/<owner>/<repo>" (no trailing slash). Empty until the
// repository is public (publication plan, AGENTS rule 6): while it is empty the bug report only saves the zip, and the
// About "Source code" button stays hidden. After publication put the address here; the bug report then opens
// <address>/issues/new.
constexpr char FORK_REPOSITORY_URL[] = "";

// The GitHub API address of the fork's latest release, "https://api.github.com/repos/<owner>/<repo>/releases/latest".
// Empty: the update check is off (the upstream KytyPS5 feeds announce upstream releases, not the fork's). Fill it in
// together with FORK_REPOSITORY_URL once the fork publishes releases whose tags match KYTY_RELEASE_TAG.
constexpr char FORK_RELEASES_FEED_URL[] = "";

// The game and game versions the fork is tested and tuned on (Play checks: "Game version ... · tested"; any other
// version or game gets a yellow "untested" note, PLAY stays on). When the fork moves to a new game patch, add the
// version here (as written in the game's sce_sys/param.json "appVersion" / "contentVersion"); the first is the main one.
constexpr char        TESTED_TITLE_ID[]       = "PPSA26344";
constexpr char        TESTED_GAME_NAME[]      = "Ghost of Yōtei";
constexpr const char* TESTED_GAME_VERSIONS[] = {"01.512.000"};

#endif // LAUNCHER_INCLUDE_FORK_INFO_H_
