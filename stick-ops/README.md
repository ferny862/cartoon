# Stick Ops

A first-person 3D stick-figure shooter styled after modern Call of Duty
multiplayer: 6 v 6 Team Deathmatch against bots in a ruined city at dusk.

Open `index.html` in a desktop browser with hardware acceleration on. It loads
Three.js from cdnjs, so it needs an internet connection. It is played with a
keyboard and mouse; the mouse is captured when you deploy, and Esc releases it.

`classic.html` is the original 2D side-scrolling version.

## What's in it

- **Map:** a ruined city block with a central tower, a house and a sniper
  platform you can climb by stairs, plus containers, cars, barriers and
  sandbags for cover.
- **Loadouts:** M4A1, MP5, KAR98K (with scope overlay) or Model 680 shotgun,
  plus an X12 sidearm and frag grenades you can cook.
- **First-person feel:** aim down sights with field-of-view zoom, recoil, weapon
  bob and sway, tactical sprint, slide, crouch and jump.
- **HUD:** rotating minimap with red pings when enemies fire, compass bar, team
  score bars and match clock, kill feed, ammo counter, hitmarkers (red on
  kills), damage direction arcs, grenade warnings, and a scoreboard on Tab.
- **Killcam:** after you die, you see the fight from your killer's eyes.
- **Killstreaks:** UAV (3 kills), Airstrike (5; aim where it should land and
  click) and Attack Helo (7). Enemy bots earn and call them too.
- **Medals:** First Blood, Headshot, Longshot, Double Kill and more, plus
  level-ups. XP is saved in the browser.
- **Bots:** they find paths around the map (including stairs to the rooftops),
  fight at their weapon's best range, flank, retreat when wounded, and throw
  grenades. Three difficulty levels.
- **Audio:** sound effects are generated in the browser and positioned in 3D,
  with spoken announcements where the browser supports them.

## Controls

| Key | Action |
|---|---|
| W A S D | Move |
| Mouse | Look · left click fire · right click aim down sights |
| Space | Jump |
| Shift | Sprint |
| C / Ctrl | Crouch (slide while sprinting) |
| R | Reload |
| Q / mouse wheel | Swap weapon |
| G (hold) | Cook and throw frag |
| V / F | Melee |
| 3 / 4 / 5 | UAV · Airstrike · Attack helo |
| Tab | Scoreboard |
| M | Mute |
| Esc | Pause |
