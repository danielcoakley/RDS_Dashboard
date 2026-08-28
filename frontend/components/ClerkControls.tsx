"use client";

import { OrganizationSwitcher, UserButton } from "@clerk/nextjs";

type ClerkControlsProps = {
  enabled: boolean;
};

export function ClerkWorkspaceControls({ enabled }: ClerkControlsProps) {
  if (!enabled) {
    return null;
  }

  return (
    <div className="clerkWorkspaceControls">
      <OrganizationSwitcher
        hidePersonal
        afterCreateOrganizationUrl="/dashboard"
        afterSelectOrganizationUrl="/dashboard"
      />
      <UserButton />
    </div>
  );
}

export function ClerkOrganizationSelector({ enabled }: ClerkControlsProps) {
  if (!enabled) {
    return null;
  }

  return (
    <div className="clerkOrganizationSelector">
      <OrganizationSwitcher
        hidePersonal
        defaultOpen
        afterCreateOrganizationUrl="/dashboard"
        afterSelectOrganizationUrl="/dashboard"
      />
    </div>
  );
}
