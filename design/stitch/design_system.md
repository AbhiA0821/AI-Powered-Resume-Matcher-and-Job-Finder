# Stitch Design Exploration: HireAgent AI Engineering Platform

## Stitch MCP Availability
- StitchMCP was queried via `call_mcp_tool(ServerName="StitchMCP", ToolName="list_projects")`.
- Active project design themes inspected: `MedIntel` and `MyAI Portfolio Premium Redesign` (Theme: Obsidian Kinetic / Deep Space).
- Extracted design principles translated into HireAgent custom design system (hand-authored, zero React/Tailwind dependencies).

## Design System Reference (Dark, Professional AI Engineering Platform)
### Colors
- Base Background: `#0A0F1E`
- Surfaces: `#111A2E` (Surface 1), `#18233B` (Surface 2), `#222D47` (Surface 3)
- Borders: `#2A3756` (Subtle), `#3A4A70` (Strong)
- Typography:
  - Text Primary: `#F1F5F9`
  - Text Secondary: `#CBD5E1`
  - Text Muted: `#94A3B8` (Contrast >= 4.5:1 on surfaces)
  - Text Disabled: `#64748B`
  - Input Placeholder: `#8FA0BA` (Contrast >= 4.5:1 on input background `#0D1527`)
- Accents:
  - Indigo Accent: `#4F46E5`, Hover `#4338CA`
  - Accent Text: `#A5B4FC`
  - AI Cyan: `#22D3EE` (Score pills, tool chips, active nav indicator, in-progress stages)
- Status & Indicators:
  - Success: `#34D399` (tint `rgba(52, 211, 153, 0.12)`)
  - Warning: `#FBBF24` (tint `rgba(251, 191, 36, 0.12)`)
  - Danger: `#F87171` (tint `rgba(248, 113, 113, 0.12)`)
### Typography
- Body / Headlines: Inter or system stack
- Numerical, Scores & Chips: JetBrains Mono or monospace stack
